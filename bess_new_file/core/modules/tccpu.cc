#include "tccpu.h"

const Commands TCCPU_ACL::cmds = {};
CommandResponse TCCPU_ACL::Init(const bess::pb::EmptyArg &) {
  model = torch::jit::load("../../model/first_layer.pt");
  assert(module != nullptr);
  turn = 0;
  log.open("torchlog",std::ios::out);
  return CommandSuccess();
}


static void build_input(double dst[], be32_t sip, be32_t dip,
                           be16_t sport, be16_t dport, uint8_t prot){
  dst[0] = (sip&be32_t(0xff)).value();
  dst[1] = ((sip&be32_t(0xff00)) >> 8).value();
  dst[2] = ((sip&be32_t(0xff0000)) >> 16).value();
  dst[3] = ((sip&be32_t(0xff000000)) >> 24).value();
  dst[4] = (dip&be32_t(0xff)).value();
  dst[5] = ((dip&be32_t(0xff00)) >> 8).value();
  dst[6] = ((dip&be32_t(0xff0000)) >> 16).value();
  dst[7] = ((dip&be32_t(0xff000000)) >> 24).value();
  dst[8] = (sport&be16_t(0xff)).value();
  dst[9] = ((sport&be16_t(0xff00)) >> 8).value();
  dst[10] = (dport&be16_t(0xff)).value();
  dst[11] = ((dport&be16_t(0xff00)) >> 8).value();
  dst[12] = prot;
}

void TCCPU_ACL::ProcessBatch(Context *ctx, bess::PacketBatch *batch) {
  using bess::utils::Ethernet;
  using bess::utils::Ipv4;
  using bess::utils::Udp;

  gate_idx_t incoming_gate = ctx->current_igate;

  int cnt = batch->cnt();
  std::vector<torch::jit::IValue> inputs;

  for (int i = 0; i < cnt; i++) {
    bess::Packet *pkt = batch->pkts()[i];
    Ethernet *eth = pkt->head_data<Ethernet *>();
    Ipv4 *ip = reinterpret_cast<Ipv4 *>(eth + 1);
    size_t ip_bytes = ip->header_length << 2;
    Udp *udp = reinterpret_cast<Udp *>(reinterpret_cast<uint8_t *>(ip) + ip_bytes);

    build_input(&vecs[13*i+13*32*turn],ip->src, ip->dst,
                udp->src_port, udp->dst_port, ip->protocol);
  }

  turn = (turn + 1) % kMod;
  if (turn == 0) {
    at::Tensor trans = torch::from_blob(vecs, {cnt*kMod,13}).to(torch::kCUDA);

    inputs.push_back(trans);
    at::Tensor output = model.forward(inputs).toTensor();

    at::Tensor label = at::argmax(output, /*dim=*/1);
  }
  for (int i = 0;i < cnt; i++) EmitPacket(ctx, batch->pkts()[i], incoming_gate);

/*
  auto labelAccessor = label.accessor<long,1>();
  for (int i = 0; i < cnt; i++) {
    if (labelAccessor[i]) {
      EmitPacket(ctx, batch->pkts()[i], incoming_gate);
    } else {
      DropPacket(ctx, batch->pkts()[i]);
    }
  }
*/
}

ADD_MODULE(TCCPU_ACL, "torch_cpu_acl", "ACL module with NN and torch jit")
