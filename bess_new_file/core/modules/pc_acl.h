#ifndef BESS_MODULES_PC_ACL_H_
#define BESS_MODULES_PC_ACL_H_

#include <vector>

#include "../module.h"
#include "../pb/module_msg.pb.h"
#include "../utils/ip.h"

using bess::utils::be16_t;
using bess::utils::be32_t;
using bess::utils::Ipv4Prefix;

class PC_ACL final : public Module {
 public:
  struct ACLRule {
    bool Match(be32_t sip, be32_t dip, be16_t sport, be16_t dport, uint8_t prot) const {
      return src_ip.Match(sip) && dst_ip.Match(dip) &&
             sport >= src_port_b && sport <= src_port_e &&
             dport >= dst_port_b && dport <= dst_port_e &&
             prot >= prot_b && prot <= prot_e;
    }

    Ipv4Prefix src_ip;
    Ipv4Prefix dst_ip;
    be16_t src_port_b,src_port_e;
    be16_t dst_port_b,dst_port_e;
    uint8_t prot_b,prot_e;
    bool drop;
  };

  static const Commands cmds;

  PC_ACL() : Module() { max_allowed_workers_ = Worker::kMaxWorkers; }

  CommandResponse Init(const bess::pb::PC_ACLArg &arg);

  void ProcessBatch(Context *ctx, bess::PacketBatch *batch) override;

  CommandResponse CommandAdd(const bess::pb::PC_ACLArg &arg);
  CommandResponse CommandClear(const bess::pb::EmptyArg &arg);

 private:
  std::vector<ACLRule> rules_;
};

#endif  // BESS_MODULES_PC_ACL_H_
