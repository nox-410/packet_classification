#include "tcgpu.h"

const Commands TCGPU_ACL::cmds = {};
CommandResponse TCGPU_ACL::Init(const bess::pb::TCGPU_ACLArg &arg) {
  std::string prefix = "../../model/";
  model = torch::jit::load(prefix+"first_layer.pt");
  assert(module != nullptr);
  turn = 0;
  log.open("torchlog",std::ios::out);


  kMod = arg.batch();
  vecs = new double [32*13*kMod];
  std::stringstream ss;
  for (int i = 1;i <= kL1Width; i++) {
    ss << prefix << "second_layer_" << i << ".pt";
    if (fopen(ss.str().data(),"r") == nullptr) {
      numL2Model = i - 1;
      break;
    }
    modelL2[i] = torch::jit::load(ss.str());
    ss.str("");
  }
  log << "Load L2 Models Number:" << numL2Model << std::endl;
  return CommandSuccess();
}

TCGPU_ACL::~TCGPU_ACL() {
  delete [] vecs;
}

static void build_input(double dst[], const ipv4_5tuple &tuple){
  auto pt = reinterpret_cast<const uint8_t*>(&tuple);
  for (int i = 0;i < 13;i++,pt++) {
    dst[i] = static_cast<double>(*pt);
  }
}

static void build_input_l2(uint32_t dst[], std::vector<ipv4_5tuple> &v){
  int n = v.size();
  for (int i = 0;i < n;i++) {
    dst[5*i] = v[i].ip_src;
    dst[5*i+1] = v[i].ip_dst;
    dst[5*i+2] = v[i].port_src;
    dst[5*i+3] = v[i].port_dst;
    dst[5*i+4] = v[i].proto;
  }
}

void TCGPU_ACL::ProcessBatch(Context *ctx, bess::PacketBatch *batch) {
  using bess::utils::Ethernet;
  using bess::utils::Ipv4;
  using bess::utils::Udp;

  gate_idx_t incoming_gate = ctx->current_igate;

  int cnt = batch->cnt();

  for (int i = 0; i < cnt; i++) {
    bess::Packet *pkt = batch->pkts()[i];
    Ethernet *eth = pkt->head_data<Ethernet *>();
    Ipv4 *ip = reinterpret_cast<Ipv4 *>(eth + 1);
    size_t ip_bytes = ip->header_length << 2;
    Udp *udp = reinterpret_cast<Udp *>(reinterpret_cast<uint8_t *>(ip) + ip_bytes);
    
    ipv4_5tuple tuple;
    tuple.proto = ip->protocol;
    tuple.ip_src = ip->src.raw_value();
    tuple.ip_dst = ip->dst.raw_value();
    tuple.port_src = udp->src_port.raw_value();
    tuple.port_dst = udp->dst_port.raw_value();
    L1Buffer.push_back(tuple);

    build_input(&vecs[13*i+13*32*turn],tuple);
  }
  turn = (turn + 1) % kMod;
  if (turn == 0) {
    std::vector<torch::jit::IValue> inputs;
    at::Tensor trans = torch::from_blob(vecs, {cnt*kMod,13}).to(torch::kCUDA);
    inputs.push_back(trans);

    at::Tensor output = model.forward(inputs).toTensor();

    at::Tensor _label = at::argmax(output, /*dim=*/1);
    at::Tensor label = _label.to(torch::kCPU);
	
    auto labelAccessor = label.accessor<long,1>();
    for (int i = 0; i < cnt*kMod; i++) {
      if (labelAccessor[i] > 0 && labelAccessor[i] <= numL2Model) {
        L2Buffer[labelAccessor[i]].push_back(L1Buffer[i]);
      } else {
        /* Handle error here */;
      }
    }

    L1Buffer.clear();

    /* forward in the second layer */
    for (int i = 0; i <= kL1Width; i++) {
      if (L2Buffer[i].empty())
        continue;
      int numItem = L2Buffer[i].size();
      uint32_t* vecs2 = new uint32_t [5 * numItem];
      build_input_l2(vecs2,L2Buffer[i]);

      trans = torch::from_blob(vecs, {numItem,5}).to(torch::kCUDA);
      inputs.clear();
      inputs.push_back(trans);
      output = modelL2[i].forward(inputs).toTensor();

      L2Buffer[i].clear();
      delete [] vecs2;
    }
  }
  for (int i = 0;i < cnt; i++) EmitPacket(ctx, batch->pkts()[i], incoming_gate);

}

ADD_MODULE(TCGPU_ACL, "torch_gpu_acl", "ACL module with NN and torch jit")
