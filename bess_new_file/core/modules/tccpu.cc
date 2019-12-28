#include "tccpu.h"
#include <torch/script.h>
#include <torch/torch.h>

#include "../utils/ether.h"
#include "../utils/ip.h"
#include "../utils/udp.h"

const Commands TCCPU_ACL::cmds = {};
CommandResponse TCCPU_ACL::Init(const bess::pb::EmptyArg &) {
    model = torch::jit::load("/home/sa/Desktop/model.pt");
    assert(module != nullptr);
  return CommandSuccess();
}


static double* build_input(be32_t sip, be32_t dip, be16_t sport, be16_t dport, uint8_t prot)
{
	double* input = new double[13];
	input[0] = (sip&be32_t(0xff)).value();
	input[1] = ((sip&be32_t(0xff00)) >> 8).value();
	input[2] = ((sip&be32_t(0xff0000)) >> 16).value();
	input[3] = ((sip&be32_t(0xff000000)) >> 24).value();
	input[4] = (dip&be32_t(0xff)).value();
	input[5] = ((dip&be32_t(0xff00)) >> 8).value();
	input[6] = ((dip&be32_t(0xff0000)) >> 16).value();
	input[7] = ((dip&be32_t(0xff000000)) >> 24).value();
	input[8] = (sport&be16_t(0xff)).value();
	input[9] = ((sport&be16_t(0xff00)) >> 8).value();
	input[10] = (dport&be16_t(0xff)).value();
	input[11] = ((dport&be16_t(0xff00)) >> 8).value();
	input[12] = prot;
	return input;
}

void TCCPU_ACL::ProcessBatch(Context *ctx, bess::PacketBatch *batch) {
  using bess::utils::Ethernet;
  using bess::utils::Ipv4;
  using bess::utils::Udp;

  gate_idx_t incoming_gate = ctx->current_igate;

  int cnt = batch->cnt();
	//double** vecs = new double*[cnt];
  //std::vector<torch::jit::IValue> inputs;
  for (int i = 0; i < cnt; i++) {
    bess::Packet *pkt = batch->pkts()[i];
std::vector<torch::jit::IValue> inputs;
    Ethernet *eth = pkt->head_data<Ethernet *>();
    Ipv4 *ip = reinterpret_cast<Ipv4 *>(eth + 1);
    size_t ip_bytes = ip->header_length << 2;
    Udp *udp =
        reinterpret_cast<Udp *>(reinterpret_cast<uint8_t *>(ip) + ip_bytes);
		double* vecs = build_input(ip->src, ip->dst, udp->src_port, udp->dst_port, ip->protocol);
		//vecs[i] = build_input(ip->src, ip->dst, udp->src_port, udp->dst_port, ip->protocol);
		auto t = torch::from_blob(vecs, {13});
    inputs.push_back(t);
at::Tensor output = model.forward(inputs).toTensor();
auto label = at::argmax(output);
if (label.item<int>())
	    EmitPacket(ctx, batch->pkts()[i], incoming_gate);
		else
      DropPacket(ctx, batch->pkts()[i]);
delete [] vecs;
  }
  /*at::Tensor output = model.forward(inputs).toTensor();
  auto label = at::argmax(output, 1)[0];

	for (int i = 0; i < cnt; i++)
	{
		if (label[i].item<int>())
	    EmitPacket(ctx, batch->pkts()[i], incoming_gate);
		else
      DropPacket(ctx, batch->pkts()[i]);
    delete [] vecs[i];
  }*/
  //delete [] vecs;
}

ADD_MODULE(TCCPU_ACL, "torch_cpu_acl", "ACL module with NN and torch jit")
