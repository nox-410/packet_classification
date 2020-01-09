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

class TCCPU_ACL final : public Module {
 public:


  static const Commands cmds;

  TCCPU_ACL() : Module() { max_allowed_workers_ = Worker::kMaxWorkers; }

  CommandResponse Init(const bess::pb::EmptyArg &arg);

  void ProcessBatch(Context *ctx, bess::PacketBatch *batch) override;
  
 private:
  torch::jit::script::Module model;

  std::ofstream log;
  
  int turn;
  const static int kMod = 1000;
  double vecs[32*13*kMod];

};

#endif  // BESS_MODULES_TCCPU_ACL_H_
