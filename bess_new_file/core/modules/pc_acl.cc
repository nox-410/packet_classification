#include "pc_acl.h"

#include "../utils/ether.h"
#include "../utils/ip.h"
#include "../utils/udp.h"

const Commands PC_ACL::cmds = {
    {"add", "PC_ACLArg", MODULE_CMD_FUNC(&PC_ACL::CommandAdd),
     Command::THREAD_UNSAFE},
    {"clear", "EmptyArg", MODULE_CMD_FUNC(&PC_ACL::CommandClear),
     Command::THREAD_UNSAFE}};

CommandResponse PC_ACL::Init(const bess::pb::PC_ACLArg &arg) {
  printf("%d\n",arg.rules_size());
  for (const auto &rule : arg.rules()) {
    ACLRule new_rule = {
      .src_ip = Ipv4Prefix(rule.src_ip()),
      .dst_ip = Ipv4Prefix(rule.dst_ip()),
      .src_port_b = be16_t(static_cast<uint16_t>(rule.src_port_b())),
      .src_port_e = be16_t(static_cast<uint16_t>(rule.src_port_e())),
      .dst_port_b = be16_t(static_cast<uint16_t>(rule.dst_port_b())),
      .dst_port_e = be16_t(static_cast<uint16_t>(rule.dst_port_e())),
      .prot_b = static_cast<uint8_t>(rule.prot_b()),
      .prot_e = static_cast<uint8_t>(rule.prot_e()),
      .drop = rule.drop()};
    rules_.push_back(new_rule);
  }
  return CommandSuccess();
}

CommandResponse PC_ACL::CommandAdd(const bess::pb::PC_ACLArg &arg) {
  Init(arg);
  return CommandSuccess();
}

CommandResponse PC_ACL::CommandClear(const bess::pb::EmptyArg &) {
  rules_.clear();
  return CommandSuccess();
}

void PC_ACL::ProcessBatch(Context *ctx, bess::PacketBatch *batch) {
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

    bool emitted = false;
    for (const auto &rule : rules_) {
      if (rule.Match(ip->src, ip->dst, udp->src_port, udp->dst_port, ip->protocol)) {
        if (!rule.drop) {
          emitted = true;
          EmitPacket(ctx, pkt, incoming_gate);
        }
        break;  // Stop matching other rules
      }
    }

    if (!emitted) {
      DropPacket(ctx, pkt);
    }
  }
}

ADD_MODULE(PC_ACL, "pc_acl", "ACL module with five field")
