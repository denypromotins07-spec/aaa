/*
 * eBPF Kill-Switch Module for NEXUS-OMEGA
 * 
 * This eBPF program attaches to the Linux networking stack and drops
 * all outbound TCP/UDP packets when the kill-switch is activated.
 * 
 * SECURITY NOTES:
 * - Requires CAP_BPF and CAP_NET_ADMIN capabilities to load
 * - Must be compiled with clang targeting BPF
 * - The shared map 'kill_switch_active' is updated from userspace
 * 
 * COMPILATION:
 *   clang -O2 -target bpf -c kill_switch_ebpf.c -o kill_switch_ebpf.o
 * 
 * LOADING (requires root):
 *   bpftool prog load kill_switch_ebpf.o /sys/fs/bpf/nexus_killswitch type sock_filter
 */

#include <linux/bpf.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>

/* Map definition for kill-switch state
 * Key: 0 (single global flag)
 * Value: u32 (0 = inactive, 1 = active/drop all)
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} kill_switch_map SEC(".maps");

/* Statistics counters */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, __u32);
    __type(value, __u64);
} stats_map SEC(".maps");

#define STAT_PACKETS_SEEN   0
#define STAT_PACKETS_DROPPED 1
#define STAT_TCP_DROPPED     2
#define STAT_UDP_DROPPED     3

/* Helper macro to increment a stat counter */
static __always_inline void increment_stat(__u32 stat_idx) {
    __u64 *counter = bpf_map_lookup_elem(&stats_map, &stat_idx);
    if (counter) {
        __sync_fetch_and_add(counter, 1);
    }
}

/* Main packet filter program
 * 
 * Returns:
 *   -1 (TC_ACT_SHOT) to drop packet
 *    0 (TC_ACT_OK) to allow packet
 */
SEC("tc")
int nexus_killswitch(struct __sk_buff *skb) {
    /* Increment packets seen counter */
    increment_stat(STAT_PACKETS_SEEN);
    
    /* Check kill-switch state */
    __u32 key = 0;
    __u32 *kill_active = bpf_map_lookup_elem(&kill_switch_map, &key);
    
    /* If map lookup fails or kill-switch not active, allow packet */
    if (!kill_active || *kill_active == 0) {
        return TC_ACT_OK;
    }
    
    /* Kill-switch is active - determine packet type for statistics */
    __u16 proto = skb->protocol;
    
    /* Handle IPv4 packets */
    if (proto == htons(ETH_P_IP)) {
        struct iphdr ip_hdr;
        
        /* Load IP header */
        if (bpf_skb_load_bytes(skb, sizeof(struct ethhdr), &ip_hdr, sizeof(ip_hdr)) < 0) {
            return TC_ACT_OK; /* Allow on error to prevent bricking */
        }
        
        /* Check for TCP */
        if (ip_hdr.protocol == IPPROTO_TCP) {
            increment_stat(STAT_TCP_DROPPED);
            increment_stat(STAT_PACKETS_DROPPED);
            return TC_ACT_SHOT; /* Drop TCP */
        }
        
        /* Check for UDP */
        if (ip_hdr.protocol == IPPROTO_UDP) {
            increment_stat(STAT_UDP_DROPPED);
            increment_stat(STAT_PACKETS_DROPPED);
            return TC_ACT_SHOT; /* Drop UDP */
        }
    }
    
    /* Handle IPv6 packets (basic check) */
    if (proto == htons(ETH_P_IPV6)) {
        /* For IPv6, we drop everything when kill-switch is active
         * More sophisticated filtering would parse the IPv6 header
         */
        increment_stat(STAT_PACKETS_DROPPED);
        return TC_ACT_SHOT;
    }
    
    /* Allow non-TCP/UDP traffic (ARP, ICMP, etc.) for network management */
    return TC_ACT_OK;
}

/* XDP variant for even faster packet dropping at the driver level
 * 
 * This can be attached to specific network interfaces using:
 *   bpftool prog load ... type xdp
 *   ip link set dev <iface> xdp obj ...
 */
SEC("xdp")
int nexus_killswitch_xdp(struct xdp_md *ctx) {
    /* Check kill-switch state */
    __u32 key = 0;
    __u32 *kill_active = bpf_map_lookup_elem(&kill_switch_map, &key);
    
    if (!kill_active || *kill_active == 0) {
        return XDP_PASS;
    }
    
    /* Kill-switch active - drop everything */
    increment_stat(STAT_PACKETS_SEEN);
    increment_stat(STAT_PACKETS_DROPPED);
    
    return XDP_DROP;
}

/* License declaration required for eBPF programs */
char LICENSE[] SEC("license") = "GPL";

/* Version information for userspace loader */
__u32 VERSION SEC("version") = 1;
