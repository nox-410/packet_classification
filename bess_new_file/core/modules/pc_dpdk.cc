#include "pc_dpdk.h"

#include "../utils/ether.h"
#include "../utils/ip.h"
#include "../utils/udp.h"

#include <arpa/inet.h>

const Commands PC_DPDK::cmds = {
    {"add", "PC_DPDKArg", MODULE_CMD_FUNC(&PC_DPDK::CommandAdd),
     Command::THREAD_UNSAFE},
    {"clear", "EmptyArg", MODULE_CMD_FUNC(&PC_DPDK::CommandClear),
     Command::THREAD_UNSAFE}};

CommandResponse PC_DPDK::Init(const bess::pb::PC_DPDKArg &arg) {
  if ((acx = rte_acl_create(&prm)) == NULL)
    return CommandFailure(EINVAL,"context create failure");
  int total = arg.rules_size();
  int prio = total;
  uint32_t i = 0;

  struct acl_ipv4_rule rules[total];

  for (const auto &rule : arg.rules()) {
    std::string src_ip = rule.src_ip();
    size_t pt = src_ip.find('/');
    uint32_t src_addr = inet_addr(src_ip.substr(0,pt).c_str());
    uint8_t src_masklen = (uint8_t)std::stoi(src_ip.substr(pt+1));

    std::string dst_ip = rule.dst_ip();
    pt = dst_ip.find('/');
    uint32_t dst_addr = inet_addr(dst_ip.substr(0,pt).c_str());
    uint8_t dst_masklen = (uint8_t)std::stoi(dst_ip.substr(pt+1));
    acl_ipv4_rule temp = {
      .data = {.category_mask = 1, .priority = prio, .userdata = i + 1},
      .field = {
        /*   protocal  */
      {.value={.u32=rule.prot_b()},.mask_range={.u32=rule.prot_e()}},
        /* source IPv4 */
      {.value={.u32=ntohl(src_addr)},.mask_range={.u8=src_masklen}},
        /* destination IPv4 */
      {.value={.u32=ntohl(dst_addr)},.mask_range={.u8=dst_masklen}},
        /* source port */
      {.value={.u32=rule.src_port_b()},.mask_range={.u32=rule.src_port_e()}},
        /* destination port */
      {.value={.u32=rule.dst_port_b()},.mask_range={.u32=rule.dst_port_e()}}}
    };

	rules[i] = temp;
	i++,prio--;
  }
  int ret = rte_acl_add_rules(acx, reinterpret_cast<rte_acl_rule *>(rules),RTE_DIM(rules));
  if (ret != 0)
    return CommandFailure(EINVAL,"error at adding ACL rules.");


  cfg.num_categories = 1;
  cfg.num_fields = RTE_DIM(ipv4_defs);
  memcpy(cfg.defs, ipv4_defs, sizeof (ipv4_defs));
  ret = rte_acl_build(acx, &cfg);
  if (ret != 0)
    return CommandFailure(EINVAL,"error at build runtime structures for ACL context.");
  return CommandSuccess();
}


CommandResponse PC_DPDK::CommandAdd(const bess::pb::PC_DPDKArg &arg) {
  Init(arg);
  return CommandSuccess();
}


CommandResponse PC_DPDK::CommandClear(const bess::pb::EmptyArg &) {
  rte_acl_reset_rules(acx);
  return CommandSuccess();
}

void PC_DPDK::ProcessBatch(Context *ctx, bess::PacketBatch *batch) {
  using bess::utils::Ethernet;
  using bess::utils::Ipv4;
  using bess::utils::Udp;

  gate_idx_t incoming_gate = ctx->current_igate;

  int cnt = batch->cnt();
  uint32_t result[cnt];
  ipv4_5tuple data[cnt];
  uint8_t *datap[cnt];
  for (int i = 0; i < cnt; i++) {
    bess::Packet *pkt = batch->pkts()[i];

    Ethernet *eth = pkt->head_data<Ethernet *>();
    Ipv4 *ip = reinterpret_cast<Ipv4 *>(eth + 1);
    size_t ip_bytes = ip->header_length << 2;
    Udp *udp = reinterpret_cast<Udp *>(reinterpret_cast<uint8_t *>(ip) + ip_bytes);

    datap[i] = reinterpret_cast<uint8_t*>(&data[i]);
    data[i].proto = ip->protocol;
    data[i].ip_src = ip->src.raw_value();
    data[i].ip_dst = ip->dst.raw_value();
    data[i].port_src = udp->src_port.raw_value();
    data[i].port_dst = udp->dst_port.raw_value();
  }
  rte_acl_classify(acx,(const uint8_t **)datap, result, cnt, 1);
  for (int i = 0;i < cnt; i++) {
    bess::Packet *pkt = batch->pkts()[i];
    if (result[i] == 0) DropPacket(ctx,pkt);
    else EmitPacket(ctx,pkt,incoming_gate);
  }
}

ADD_MODULE(PC_DPDK, "pc_dpdk", "DPDK ACL module")
