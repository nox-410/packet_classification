#ifndef BESS_MODULES_TCCPU_ACL_H_
#define BESS_MODULES_TCCPU_ACL_H_

#include <torch/script.h>
#include <torch/torch.h>

#include "../module.h"
#include "../pb/module_msg.pb.h"
#include "../utils/ether.h"
#include "../utils/ip.h"
#include "../utils/udp.h"

#include <vector>
#include <fstream>
using bess::utils::be16_t;
using bess::utils::be32_t;
using bess::utils::Ipv4Prefix;

struct ipv4_5tuple {
  uint32_t ip_src;
  uint32_t ip_dst;
  uint16_t port_src;
  uint16_t port_dst;
  uint8_t proto;
};

class TCCPU_ACL final : public Module {
 public:


  static const Commands cmds;

  TCCPU_ACL() : Module() { max_allowed_workers_ = Worker::kMaxWorkers; }

  CommandResponse Init(const bess::pb::EmptyArg &arg);

  void ProcessBatch(Context *ctx, bess::PacketBatch *batch) override;
  
 private:

  const static int kL1Width = 256;
  const static int kMod = 1000;
  torch::jit::script::Module model;
  torch::jit::script::Module modelL2[kL1Width + 1];
  //start from index 1

  std::vector<ipv4_5tuple> L2Buffer[kL1Width + 1];
  std::vector<ipv4_5tuple> L1Buffer;

  std::ofstream log;
  
  int turn;
  int numL2Model;
  double vecs[32*13*kMod];
};

#endif  // BESS_MODULES_TCCPU_ACL_H_
