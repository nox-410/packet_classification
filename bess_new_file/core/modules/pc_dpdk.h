#ifndef BESS_MODULES_PC_DPDK_H_
#define BESS_MODULES_PC_DPDK_H_

#include <cstring>
#include <iostream>
#include <rte_config.h>
#include <rte_acl.h>
#include <rte_eal.h>
#include "../module.h"
#include "../pb/module_msg.pb.h"
#include "../utils/ip.h"

using bess::utils::be16_t;
using bess::utils::be32_t;
using bess::utils::Ipv4Prefix;

struct ipv4_5tuple {
  uint8_t proto;
  uint32_t ip_src;
  uint32_t ip_dst;
  uint16_t port_src;
  uint16_t port_dst;
};

struct rte_acl_field_def ipv4_defs[5] = {
  /* first input field protocal - always one byte long. */
  {
    .type = RTE_ACL_FIELD_TYPE_RANGE,
    .size = sizeof (uint8_t),
    .field_index = 0,
    .input_index = 0,
    .offset = offsetof (struct ipv4_5tuple, proto),
  },
  /* next input field (IPv4 src address) - 4 consecutive bytes. */
  {
    .type = RTE_ACL_FIELD_TYPE_MASK,
    .size = sizeof (uint32_t),
    .field_index = 1,
    .input_index = 1,
    .offset = offsetof (struct ipv4_5tuple, ip_src),
  },
  /* next input field (IPv4 dst address) - 4 consecutive bytes. */
  {
    .type = RTE_ACL_FIELD_TYPE_MASK,
    .size = sizeof (uint32_t),
    .field_index = 2,
    .input_index = 2,
    .offset = offsetof (struct ipv4_5tuple, ip_dst),
  },
  /*
   * Next 2 fields (src & dst ports) form 4 consecutive bytes.
   * They share the same input index.
   */
  {
    .type = RTE_ACL_FIELD_TYPE_RANGE,
    .size = sizeof (uint16_t),
    .field_index = 3,
    .input_index = 3,
    .offset = offsetof (struct ipv4_5tuple, port_src),
  },
  {
    .type = RTE_ACL_FIELD_TYPE_RANGE,
    .size = sizeof (uint16_t),
    .field_index = 4,
    .input_index = 3,
    .offset = offsetof (struct ipv4_5tuple, port_dst),
  },
};

/* define a structure for the rule with up to 5 fields. */

RTE_ACL_RULE_DEF(acl_ipv4_rule, RTE_DIM(ipv4_defs));

/* AC context creation parameters. */

struct rte_acl_param prm = {
  .name = "ACL_DPDK",
  .socket_id = SOCKET_ID_ANY,
  .rule_size = RTE_ACL_RULE_SZ(RTE_DIM(ipv4_defs)),
  .max_rule_num = 50000, /* maximum number of rules in the AC context. */
};

class PC_DPDK final : public Module {
 public:
  static const Commands cmds;

  PC_DPDK() : Module() { max_allowed_workers_ = Worker::kMaxWorkers; }

  CommandResponse Init(const bess::pb::PC_DPDKArg &arg);

  void ProcessBatch(Context *ctx, bess::PacketBatch *batch) override;

  CommandResponse CommandAdd(const bess::pb::PC_DPDKArg &arg);
  CommandResponse CommandClear(const bess::pb::EmptyArg &arg);

 private:
  struct rte_acl_ctx * acx;
  struct rte_acl_config cfg;

};

#endif  // BESS_MODULES_PC_DPDK_H_
