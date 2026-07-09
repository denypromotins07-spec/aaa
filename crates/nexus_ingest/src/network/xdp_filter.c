// Chapter 1: eBPF/XDP Filter Program for Packet Filtering
// 
// This eBPF program runs in the kernel and filters exchange heartbeat
// noise directly at the NIC driver level before packets reach userspace.
// It drops irrelevant packets to reduce CPU load on the ingestion system.

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/udp.h>
#include <linux/tcp.h>

// Define constants for packet filtering
#define EXCHANGE_HEARTBEAT_PORT 12345
#define MAX_PACKET_SIZE 9000
#define CACHE_LINE_SIZE 64

// XDP action codes
#define XDP_ABORTED 1
#define XDP_DROP 2
#define XDP_PASS 3
#define XDP_TX 4
#define XDP_REDIRECT 5

// Packet metadata structure (mirrors Rust struct)
struct packet_metadata {
    __u64 timestamp_ns;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u16 packet_len;
    __u8 exchange_id;
    __u8 message_type;
    __u8 reserved[6];
};

// Map for tracking filtered packet counts (for debugging)
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 10);
    __type(key, __u32);
    __type(value, __u64);
} stats_map SEC(".maps");

// Map for exchange port configuration
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u16);
    __type(value, __u8);
} exchange_ports SEC(".maps");

// Map for IP whitelist (allowed exchanges)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, __u32);
    __type(value, __u8);
} ip_whitelist SEC(".maps");

// Helper function to update statistics
static __always_inline void update_stat(__u32 index, __u64 increment) {
    __u64 *count = bpf_map_lookup_elem(&stats_map, &index);
    if (count) {
        *count += increment;
    }
}

// Helper function to check if port is an exchange heartbeat
static __always_inline bool is_heartbeat_port(__u16 port) {
    // Check common heartbeat ports
    if (port == EXCHANGE_HEARTBEAT_PORT) return true;
    if (port == 12346 || port == 12347 || port == 12348) return true;
    
    // Check configured heartbeat ports
    __u8 *is_heartbeat = bpf_map_lookup_elem(&exchange_ports, &port);
    return is_heartbeat != NULL && *is_heartbeat == 1;
}

// Helper function to check if IP is whitelisted
static __always_inline bool is_ip_whitelisted(__u32 ip) {
    __u8 *whitelisted = bpf_map_lookup_elem(&ip_whitelist, &ip);
    return whitelisted != NULL && *whitelisted == 1;
}

// Parse IPv4 header
static __always_inline int parse_ipv4(struct ethhdr *eth, struct iphdr **iphdr) {
    if (eth->h_proto != bpf_htons(ETH_P_IP)) {
        return -1;
    }
    
    *iphdr = (struct iphdr *)(eth + 1);
    
    // Verify minimum IP header length
    if ((*iphdr)->ihl < 5) {
        return -1;
    }
    
    return 0;
}

// Parse TCP/UDP header
static __always_inline int parse_transport(struct iphdr *iph, __u16 *src_port, __u16 *dst_port, __u8 *protocol) {
    *protocol = iph->protocol;
    
    void *transport_hdr = (void *)iph + (iph->ihl * 4);
    
    if (protocol == IPPROTO_UDP) {
        struct udphdr *udph = (struct udphdr *)transport_hdr;
        *src_port = bpf_ntohs(udph->source);
        *dst_port = bpf_ntohs(udph->dest);
    } else if (protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = (struct tcphdr *)transport_hdr;
        *src_port = bpf_ntohs(tcph->source);
        *dst_port = bpf_ntohs(tcph->dest);
    } else {
        return -1;
    }
    
    return 0;
}

// Main XDP program entry point
SEC("xdp")
int nexus_xdp_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    
    // Validate minimum packet size
    if (data + sizeof(struct ethhdr) > data_end) {
        update_stat(0, 1); // Invalid packet count
        return XDP_DROP;
    }
    
    struct ethhdr *eth = (struct ethhdr *)data;
    struct iphdr *iph = NULL;
    
    // Parse IPv4 header
    if (parse_ipv4(eth, &iph) < 0) {
        update_stat(1, 1); // Non-IPv4 packet
        return XDP_PASS; // Pass non-IP traffic
    }
    
    // Validate IP header bounds
    if ((void *)(iph + 1) > data_end) {
        update_stat(0, 1);
        return XDP_DROP;
    }
    
    // Check IP whitelist
    if (!is_ip_whitelisted(iph->saddr)) {
        update_stat(2, 1); // Non-whitelisted IP
        return XDP_DROP;
    }
    
    __u16 src_port = 0, dst_port = 0;
    __u8 protocol = 0;
    
    // Parse transport layer
    if (parse_transport(iph, &src_port, &dst_port, &protocol) < 0) {
        update_stat(3, 1); // Unknown protocol
        return XDP_PASS;
    }
    
    // Filter exchange heartbeat noise
    if (is_heartbeat_port(dst_port) || is_heartbeat_port(src_port)) {
        update_stat(4, 1); // Heartbeat dropped
        return XDP_DROP;
    }
    
    // Validate packet size
    if (ctx->data_end - ctx->data > MAX_PACKET_SIZE) {
        update_stat(5, 1); // Oversized packet
        return XDP_DROP;
    }
    
    // Packet passed all filters - forward to userspace
    update_stat(6, 1); // Packets forwarded
    return XDP_PASS;
}

// License for GPL compatibility (required for some helper functions)
char LICENSE[] SEC("license") = "GPL";

// Version information
char VERSION[] SEC("version") = "1.0.0";

// Description
char DESCRIPTION[] SEC("description") = 
    "Nexus-Omega XDP filter for exchange market data ingestion. "
    "Drops heartbeat noise and non-whitelisted traffic at the NIC level.";
