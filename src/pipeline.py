import copy
import time
import os
import glob
import json
import csv
from collections import defaultdict
from archipelago.pnr_graph import (
    KernelNodeType,
    construct_graph,
    construct_kernel_graph,
    TileType,
    RouteType,
    TileNode,
    RouteNode,
)
from archipelago.sta import sta
import pythunder
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import math
import traceback
from pathlib import Path


# def verboseprint(*args, **kwargs):
#     print(*args, **kwargs)

# Notes:
# Capstone I is Guardband
# Capstone II is Conformal
# Capstone III is Bounded
#
# With POST_PNR_ITR set, simultaneous evaluation is enabled by default. One
# shared pipelining trajectory evaluates the uncapped baseline, Capstone I,
# Capstone II, and all five Table VIII Capstone III bounds. Set
# CAPSTONE_RUN_ALL_MODES=0 to use one mode instead.
# The simultaneous version writes capstone_all_modes_trace.csv,
# capstone_all_modes_summary.csv, capstone_all_modes_bitstreams.csv,
# capstone_figure11_timing.csv, and capstone_all_modes_selection.json in the
# application directory unless those paths are overridden with environment
# variables of the same names prefixed by CAPSTONE_ALL_MODES_.
#
# Mean-power frequency/II scaling is enabled by default:
#   gamma_eff = gamma_hat * (f / II) / (f_ref / II_ref)
# with CAPSTONE_FREQ_REF_MHZ=100 and II_ref equal to the current application II.
# AHA uses pipeline_config_interval=0 for an unconstrained/default
# schedule. Capstone's power model maps this to effective II=1. To disable
# scaling, set CAPSTONE_USE_FREQ_II_SCALING=0.
#
# Capstone II reads global calibration samples from CAPSTONE_II_CALIBRATION_JSON.
# Accepted sample fields include:
#   reference_power_mW, predicted_power_mW, frequency_mhz
# The script calculates one-sided normalized residuals
#   max(0, reference - prediction) / max(1, frequency / f_ref)
# and then applies the finite-sample split-conformal quantile index. With the
# paper's n_global=24, alpha_spec=0.05 has a finite quantile but
# alpha_anchor=0.005 does not. Therefore, the default anchor policy for this 
# demonstration is "disable". Use 199+ calibration samples for the anchor guarantee,
# or explicitly choose and report a nonconformal fallback:
#   CAPSTONE_II_ANCHOR_FALLBACK_POLICY=engineering
#   CAPSTONE_II_ANCHOR_FALLBACK_Q_MW=<justified margin>
# or CAPSTONE_II_ANCHOR_FALLBACK_POLICY=empirical_max.


verboseprint = lambda *a, **k: None


class bcolors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def effective_power_ii(value):
    """
    Convert AHA's scheduling initiation interval (II) representation to a valid II for Capstone's power model.
    AHA may use pipeline_config_interval=0 to mean that no positive initiation
    interval was supplied. Throughput scaling cannot divide by zero, so this
    is interpreted as one result per cycle (effective II=1). Positive
    IIs are preserved exactly.
    """

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Power-model II must be finite, got {value}.")
    return value if value > 0.0 else 1.0


def find_break_idx(graph, crit_path):
    crit_path_adjusted = [abs(c - crit_path[-1][1] / 2) for n, c in crit_path]
    break_idx = crit_path_adjusted.index(min(crit_path_adjusted))

    if len(crit_path) < 2:
        raise ValueError("Can't find available register on critical path")

    if graph.sparse and len(crit_path) < 5:
        raise ValueError("Can't find available FIFO on critical path")

    min_path = crit_path[-1][1]
    min_idx = -1

    if graph.sparse:
        for idx, node in enumerate(crit_path[:-4]):
            if (
                isinstance(crit_path[idx][0], RouteNode)
                and crit_path[idx][0].route_type == RouteType.SB
                and crit_path[idx + 1][0].route_type == RouteType.RMUX
                and isinstance(crit_path[idx + 2][0], RouteNode)
                and crit_path[idx + 2][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 3][0], RouteNode)
                and crit_path[idx + 3][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 4][0], RouteNode)
                and crit_path[idx + 4][0].route_type == RouteType.RMUX
            ):
                if crit_path_adjusted[idx] < min_path:
                    min_path = crit_path_adjusted[idx]
                    min_idx = idx
    else:
        for idx, node in enumerate(crit_path[:-1]):
            if (
                isinstance(crit_path[idx][0], RouteNode)
                and crit_path[idx][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 1][0], RouteNode)
                and crit_path[idx + 1][0].route_type == RouteType.RMUX
            ):
                if crit_path_adjusted[idx] < min_path:
                    min_path = crit_path_adjusted[idx]
                    min_idx = idx

    if min_idx == -1:
        raise ValueError("Can't find available register on critical path")

    return min_idx


def reg_into_route(routes, g_break_node_source, new_reg_route_source):
    for net_id, net in routes.items():
        for route in net:
            for idx, segment in enumerate(route):
                if g_break_node_source.to_route() == segment:
                    route.insert(idx + 1, new_reg_route_source.to_route())
                    return
    assert (
        False
    ), f"Couldn't find segment {g_break_node_source.to_route()} in routing file"


def break_crit_path(graph, id_to_name, crit_path, placement, routes):
    break_idx = find_break_idx(graph, crit_path)

    break_node_source = crit_path[break_idx][0]
    break_node_dest = graph.sinks[break_node_source][0]

    assert isinstance(break_node_source, RouteNode)
    assert break_node_source.route_type == RouteType.SB
    assert isinstance(break_node_dest, RouteNode)
    assert break_node_dest.route_type == RouteType.RMUX

    x = break_node_source.x
    y = break_node_source.y
    track = break_node_source.track
    bw = break_node_source.bit_width
    net_id = break_node_source.net_id
    kernel = break_node_source.kernel
    side = break_node_source.side
    verboseprint("\nBreaking net:", net_id, "Kernel:", kernel)

    dir_map = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

    new_segment = ["REG", f"T{track}_{dir_map[side]}", track, x, y, bw]
    new_reg_route_source = graph.segment_to_node(new_segment, net_id, kernel)
    new_reg_route_source.reg = True
    new_reg_route_source.update_tile_id()
    new_reg_route_dest = graph.segment_to_node(new_segment, net_id, kernel)
    new_reg_tile = TileNode(x, y, tile_id=f"r{graph.added_regs}", kernel=kernel)

    new_reg_tile.input_port_latencies["reg"] = 1
    new_reg_tile.input_port_break_path["reg"] = True

    graph.added_regs += 1

    graph.edges.remove((break_node_source, break_node_dest))
    graph.add_node(new_reg_route_source)
    graph.add_node(new_reg_tile)
    graph.add_node(new_reg_route_dest)

    graph.add_edge(break_node_source, new_reg_route_source)
    graph.add_edge(new_reg_route_source, new_reg_tile)
    graph.add_edge(new_reg_tile, new_reg_route_dest)
    graph.add_edge(new_reg_route_dest, break_node_dest)

    reg_into_route(routes, break_node_source, new_reg_route_source)
    placement[new_reg_tile.tile_id] = (new_reg_tile.x, new_reg_tile.y)
    id_to_name[new_reg_tile.tile_id] = f"pnr_pipelining{graph.added_regs}"

    graph.update_sources_and_sinks()
    graph.update_edge_kernels()

    if graph.sparse:
        break_idx += 3
        break_node_source = crit_path[break_idx][0]
        break_node_dest = graph.sinks[break_node_source][0]

        assert isinstance(break_node_source, RouteNode)
        assert break_node_source.route_type == RouteType.SB
        assert isinstance(break_node_dest, RouteNode)
        assert break_node_dest.route_type == RouteType.RMUX

        x = break_node_source.x
        y = break_node_source.y
        track = break_node_source.track
        bw = break_node_source.bit_width
        net_id = break_node_source.net_id
        kernel = break_node_source.kernel
        side = break_node_source.side
        verboseprint("\nBreaking net:", net_id, "Kernel:", kernel)

        dir_map = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

        new_segment = ["REG", f"T{track}_{dir_map[side]}", track, x, y, bw]
        new_reg_route_source = graph.segment_to_node(new_segment, net_id, kernel)
        new_reg_route_source.reg = True
        new_reg_route_source.update_tile_id()
        new_reg_route_dest = graph.segment_to_node(new_segment, net_id, kernel)
        new_reg_tile = TileNode(x, y, tile_id=f"r{graph.added_regs}", kernel=kernel)

        new_reg_tile.input_port_latencies["reg"] = 1
        new_reg_tile.input_port_break_path["reg"] = True

        graph.added_regs += 1

        graph.edges.remove((break_node_source, break_node_dest))
        graph.add_node(new_reg_route_source)
        graph.add_node(new_reg_tile)
        graph.add_node(new_reg_route_dest)

        graph.add_edge(break_node_source, new_reg_route_source)
        graph.add_edge(new_reg_route_source, new_reg_tile)
        graph.add_edge(new_reg_tile, new_reg_route_dest)
        graph.add_edge(new_reg_route_dest, break_node_dest)

        reg_into_route(routes, break_node_source, new_reg_route_source)
        placement[new_reg_tile.tile_id] = (new_reg_tile.x, new_reg_tile.y)
        id_to_name[new_reg_tile.tile_id] = f"pnr_pipelining{graph.added_regs}"

        graph.update_sources_and_sinks()
        graph.update_edge_kernels()


def break_at(graph, node1, id_to_name, placement, routing):
    path = []
    curr_node = node1
    kernel = curr_node.kernel

    while len(graph.sinks[curr_node]) == 1:
        if (
            len(graph.sources[graph.sinks[curr_node][0]]) > 1
            or graph.sinks[curr_node][0].kernel != kernel
        ):
            break
        curr_node = graph.sinks[curr_node][0]

    idx = 0
    while len(graph.sources[curr_node]) == 1:
        if (
            len(graph.sinks[curr_node]) > 1
            or graph.sources[curr_node][0].kernel != kernel
        ):
            break
        path.append((curr_node, idx))
        curr_node = graph.sources[curr_node][0]
        if curr_node in graph.get_ponds():
            break

    if curr_node in graph.get_ponds():
        verboseprint("\t\tFound pond for branch delay matching", curr_node)
        curr_node.input_port_latencies["data_in_pond"] += 1
        return

    if len(path) == 0:
        raise ValueError(f"Cant break at node: {node1}")
    path.reverse()
    ret = []
    for p in path:
        ret.append((p[0], idx))
        idx += 1
    break_crit_path(graph, id_to_name, ret, placement, routing)


def exhaustive_pipe(graph, id_to_name, placement, routing):
    for node in graph.nodes:
        if node in graph.get_tiles() or len(graph.sinks[node]) > 1:
            for sink in graph.sinks[node]:
                path = []
                curr_node = sink
                while True:
                    path.append((curr_node, len(path)))
                    if len(graph.sinks[curr_node]) != 1:
                        break
                    curr_node = graph.sinks[curr_node][0]

                for idx in range(len(path)):
                    if graph.sparse:
                        if idx + 4 >= len(path):
                            break
                        if (
                            isinstance(path[idx][0], RouteNode)
                            and path[idx][0].route_type == RouteType.SB
                            and path[idx + 1][0].route_type == RouteType.RMUX
                            and isinstance(path[idx + 2][0], RouteNode)
                            and path[idx + 2][0].route_type == RouteType.SB
                            and isinstance(path[idx + 3][0], RouteNode)
                            and path[idx + 3][0].route_type == RouteType.SB
                            and isinstance(path[idx + 4][0], RouteNode)
                            and path[idx + 4][0].route_type == RouteType.RMUX
                        ):
                            try:
                                break_crit_path(
                                    graph,
                                    id_to_name,
                                    path[idx : idx + 5],
                                    placement,
                                    routing,
                                )
                            except:
                                verboseprint("Skip")
                    else:
                        if idx + 1 >= len(path):
                            break
                        if (
                            isinstance(path[idx][0], RouteNode)
                            and path[idx][0].route_type == RouteType.SB
                            and isinstance(path[idx + 1][0], RouteNode)
                            and path[idx + 1][0].route_type == RouteType.RMUX
                        ):
                            try:
                                break_crit_path(
                                    graph,
                                    id_to_name,
                                    path[idx : idx + 2],
                                    placement,
                                    routing,
                                )
                            except:
                                verboseprint("Skip")


def add_delay_to_kernel(graph, kernel, added_delay, id_to_name, placement, routing):
    kernel_output_nodes = graph.get_output_tiles_of_kernel(kernel)
    for node in kernel_output_nodes:
        for _ in range(added_delay):
            break_at(graph, node, id_to_name, placement, routing)


def branch_delay_match_all_nodes(graph, id_to_name, placement, routing):
    nodes = graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()

        if len(graph.sources[node]) == 0:
            if node in graph.get_pes():
                cycles = {None}
            else:
                cycles = {0}

        for parent in graph.sources[node]:
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]

            if c != None and len(graph.sinks[node]) > 0 and isinstance(node, TileNode):
                c += node.input_port_latencies[parent.port]

            # Flush signals shouldn't be considered here
            if "reset" not in node.kernel:
                cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(graph.sources[node]) > 1 and len(cycles) > 1:
            verboseprint(f"Incorrect node delay: {node} {cycles}")

        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None


def find_closest_match(kernel_target, candidates):
    candidates = [c for c in candidates if "io1_" not in c]

    if "op_" + kernel_target in candidates:
        return "op_" + kernel_target

    kernel_target_out = kernel_target + "_write"
    for c in candidates:
        if kernel_target_out in c:
            return c

    kernel_target_in = kernel_target + "_read"
    kernel_target_in = kernel_target_in.replace(
        "global_wrapper_global_wrapper", "global_wrapper_glb"
    )
    for c in candidates:
        if kernel_target_in in c:
            return c

    kernel_target_in = kernel_target + "_read"
    kernel_target_in = kernel_target_in.replace("cgra", "glb")
    for c in candidates:
        if kernel_target_in in c:
            return c

    print("No match for", kernel_target)


def branch_delay_match_within_kernels(
    graph, id_to_name, placement, routing, kernel_latencies, port_remap
):
    port_remap_r = {v: k for k, v in port_remap["pe"].items()}
    port_remap_r["reg"] = "reg"
    nodes = graph.topological_sort()
    nodes.reverse()
    node_cycles = {}

    for node in nodes:
        if node.kernel not in node_cycles:
            node_cycles[node.kernel] = {}

        cycles = set()
        if len(graph.sinks[node]) == 0:
            cycles = {0}

        for sink in graph.sinks[node]:
            if sink not in node_cycles[node.kernel]:
                node_cycles[node.kernel][sink] = 0

            c = node_cycles[node.kernel][sink]

            if c != None and isinstance(sink, TileNode):
                c += sink.input_port_latencies[node.port]
            elif node in graph.get_input_ios():
                # Need special case for input IOs
                c += node.input_port_latencies["output"]

            if (
                isinstance(node, TileNode)
                and node.tile_type == TileType.PE
                and sink.port == "PondTop_output_width_17_num_0"
            ):
                continue
            cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(cycles) > 1:
            if "IO2MEM_REG_CHAIN" in os.environ or "MEM2PE_REG_CHAIN" in os.environ:
                continue
            verboseprint(
                f"\tIncorrect delay within kernel: {node.kernel} {node} {cycles}"
            )
            verboseprint(f"\tFixing branching delays at: {node} {cycles}")
            sink_cycles = [
                node_cycles[node.kernel][sink]
                for sink in graph.sinks[node]
                if node_cycles[node.kernel][sink] != None
            ]
            max_sink_cycles = max(sink_cycles)
            for sink in graph.sinks[node]:
                for _ in range(max_sink_cycles - node_cycles[node.kernel][sink]):
                    break_at(
                        graph,
                        node,
                        id_to_name,
                        placement,
                        routing,
                    )
            node_cycles[node.kernel][node] = max(cycles)
        elif len(cycles) == 1:
            node_cycles[node.kernel][node] = max(cycles)
        else:
            node_cycles[node.kernel][node] = None

    # Only certain inputs of compute kernels can have different latencies (dictated by clockwork and H2H)
    # First determine which nodes can have unique latencies
    ports_with_unique_latenices = {}
    for kernel, latency_dict in kernel_latencies.items():
        if "_glb_" in kernel:
            continue
        match = find_closest_match(kernel, list(node_cycles.keys()))
        if match is not None:
            ports_with_unique_latenices[match] = []
            for kernel_port, d1 in latency_dict.items():
                if d1["pe_port"] != []:
                    port_nodes = []
                    for compute_file_tile, compute_file_port in d1["pe_port"]:
                        found = False
                        for pe in graph.get_tiles():
                            if (
                                graph.id_to_name[str(pe)]
                                == f"{match}$inner_compute${compute_file_tile}"
                            ):
                                found_port = False
                                for source in graph.sources[pe]:
                                    if source.port in port_remap_r:
                                        port = port_remap_r[source.port]
                                        if port == compute_file_port:
                                            found = True
                                            found_port = True
                                            port_nodes.append(source)

                                for source_node, dest_node in graph.removed_edges:
                                    if dest_node == pe:
                                        if source_node.port in port_remap_r:
                                            port = port_remap_r[source_node.port]
                                            if port == compute_file_port:
                                                found = True
                                                found_port = True
                                                port_nodes.append(source_node)

                                if not found_port:
                                    print("Couldn't find pe port")
                                    print(latency_dict)
                                    breakpoint()

                        if not found:
                            print("Couldn't find pe")
                            print(latency_dict)
                            breakpoint()

                    ports_with_unique_latenices[match].append(port_nodes)

    # Then branch delay match the nodes without unique latencies
    for kernel in node_cycles:
        if kernel not in ports_with_unique_latenices:
            continue

        for nodes_with_same_latency in ports_with_unique_latenices[kernel]:
            kernel_input_latencies = [
                node_cycles[kernel][kernel_input]
                for kernel_input in nodes_with_same_latency
            ]

            for node_with_same_latency in nodes_with_same_latency:
                same_latency = max(kernel_input_latencies)

                if (
                    node_cycles[kernel][node_with_same_latency] != same_latency
                    and node_with_same_latency
                    not in ports_with_unique_latenices[kernel]
                ):
                    verboseprint(
                        f"\tIncorrect delay between ports of kernel: {kernel} {node_with_same_latency} {node_cycles[kernel][node_with_same_latency]} {same_latency}"
                    )
                    verboseprint(
                        f"\tFixing branching delays at: {node_with_same_latency}"
                    )
                    for sink in graph.sinks[node_with_same_latency]:
                        for _ in range(
                            same_latency - node_cycles[kernel][node_with_same_latency]
                        ):
                            break_at(
                                graph,
                                sink,
                                id_to_name,
                                placement,
                                routing,
                            )
                    node_cycles[node.kernel][node_with_same_latency] = same_latency

    kernel_latencies = {}
    for kernel in node_cycles:
        kernel_latencies[kernel] = max(node_cycles[kernel].values())

    return kernel_latencies, node_cycles


def branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing):
    nodes = kernel_graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()

        if len(kernel_graph.sources[node]) == 0:
            if (
                node.kernel_type == KernelNodeType.COMPUTE
                or node.kernel_type == KernelNodeType.MEM
            ):
                cycles = {None}
            else:
                cycles = {0}

        for parent in kernel_graph.sources[node]:
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]

            if c is not None:
                c += node.latency

            if not (
                "reset" in parent.kernel
                or (parent.kernel_type == KernelNodeType.MEM and str(parent)[0] == "m")
            ):
                cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(kernel_graph.sources[node]) > 1 and len(cycles) > 1:
            verboseprint(f"\tIncorrect kernel delay: {node} {cycles}")

            source_cycles = [
                node_cycles[source]
                for source in kernel_graph.sources[node]
                if node_cycles[source] != None
            ]
            max_cycle = max(source_cycles)
            for source in kernel_graph.sources[node]:
                if node_cycles[source] != None and node_cycles[source] != max_cycle:
                    verboseprint(
                        f"\tFixing kernel delays at: {source} {max_cycle - node_cycles[source]}"
                    )
                    add_delay_to_kernel(
                        graph,
                        source.kernel,
                        max_cycle - node_cycles[source],
                        id_to_name,
                        placement,
                        routing,
                    )
        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None


def flush_cycles(
    graph, id_to_name, harden_flush, pipeline_config_interval, pes_with_packed_ponds
):
    if harden_flush:
        flush_cycles = {}
        for mem in graph.get_mems() + graph.get_ponds():
            if mem.y == 0 or pipeline_config_interval == 0:
                flush_cycles[mem] = 0
            else:
                flush_cycles[mem] = (mem.y - 1) // pipeline_config_interval

        for pe in graph.get_pes():
            if (
                pes_with_packed_ponds is not None
                and pe.tile_id in pes_with_packed_ponds
            ):
                pond = pes_with_packed_ponds[pe.tile_id]
                if pe.y == 0 or pipeline_config_interval == 0:
                    flush_cycles[pond] = 0
                else:
                    flush_cycles[pond] = (pe.y - 1) // pipeline_config_interval

    else:
        for io in graph.get_input_ios():
            if io.kernel == "io1in_reset":
                break
        assert io.kernel == "io1in_reset"
        flush_cycles = {}

        for mem in graph.get_mems() + graph.get_ponds():
            for parent_node in graph.sources[mem]:
                if parent_node.port == "flush":
                    break
            if parent_node.port != "flush":
                continue

            curr_node = mem
            flush_cycles[mem] = 0
            while parent_node != io:
                if isinstance(curr_node, TileNode):
                    flush_cycles[mem] += curr_node.input_port_latencies[
                        parent_node.port
                    ]
                curr_node = parent_node
                parent_node = graph.sources[parent_node][0]

    max_flush_cycles = max(flush_cycles.values())
    for mem, flush_c in flush_cycles.items():
        flush_cycles[mem] = max_flush_cycles - flush_c

    return flush_cycles, max_flush_cycles


def find_stencil_valid_mem(graph, kernel):
    for node in graph.nodes:
        if node.kernel == kernel:
            break

    curr_node = node

    while True:
        if isinstance(curr_node, TileNode) and curr_node.tile_type == TileType.MEM:
            return curr_node

        if len(graph.sources[curr_node]) == 0:
            return None

        curr_node = graph.sources[curr_node][0]


def calculate_latencies(
    graph, kernel_graph, node_latencies, kernel_latencies, port_remap, instance_to_instr
):
    port_remap_r = {v: k for k, v in port_remap["pe"].items()}

    max_latencies = {}

    for node in kernel_graph.nodes:
        if node.kernel_type == KernelNodeType.COMPUTE:
            max_latencies[node.kernel] = node.latency

    stencil_valid_adjust = {}

    for node16 in max_latencies:
        for node1 in max_latencies:
            if (
                node16 != node1
                and node16.split("_write")[0].replace("io16", "io1")
                == node1.split("_write")[0]
            ):
                max_diff = -(max_latencies[node16] + 2)

                # Need to absorb the added latency of the stencil valids into either the compute kernel or the stencil valid schedule generator
                max_latencies[node16] -= max_latencies[node1]
                max_latencies[node1] = 0

                if max_latencies[node16] < max_diff:
                    raise Exception(
                        f"Can't absorb stencil valid latency of {max_latencies[node16]} into compute kernel"
                    )

                # if max_latencies[node16] < 0:
                stencil_valid_mem = find_stencil_valid_mem(graph, node1)
                # need to adjust stencil valid latency
                stencil_valid_adjust[stencil_valid_mem.kernel] = max_latencies[node16]
                max_latencies[node16] = 0

    # Kernel latencies are from the file passed to clockwork, need to match the latencies in the routing graph to this
    for kernel, latency_dict in kernel_latencies.items():
        # glb kernels are not in the routing graph
        if "_glb_" in kernel:
            continue

        # Find the closest matched kernel in the routing graph, we don't have exact matches because renaming
        match = find_closest_match(kernel, list(node_latencies.keys()))
        if match is not None:
            for kernel_port, d1 in latency_dict.items():
                if d1["pe_port"] == [] and match in max_latencies:
                    kernel_latencies[kernel][kernel_port][
                        "latency"
                    ] = max_latencies[match]
                elif d1["pe_port"] != []:
                    found = False
                    for compute_file_tile, compute_file_port in d1["pe_port"]:
                        # Within this loop, all the ports should have the same latency
                        found_lat = None
                        for pe in graph.get_tiles():
                            if (
                                graph.id_to_name[str(pe)]
                                == f"{match}$inner_compute${compute_file_tile}"
                            ):
                                found_port = False
                                for source in graph.sources[pe]:
                                    if source.port in port_remap_r:
                                        port = port_remap_r[source.port]
                                        if port == compute_file_port:
                                            reg = graph.get_connected_reg(source)
                                            if reg is not None:
                                                lat = node_latencies[match][reg]
                                            else:
                                                lat = node_latencies[match][source]

                                            if found_lat is not None:
                                                assert (
                                                    lat == found_lat
                                                ), f"Found multiple latencies for {kernel} {kernel_port} {compute_file_tile} {compute_file_port} {lat} {found_lat}"
                                            kernel_latencies[kernel][kernel_port]["latency"] = lat
                                            found = True
                                            found_port = True
                                            found_lat = lat
                                            break
                                if not found_port:
                                    found = True
                                    kernel_latencies[kernel][kernel_port][
                                        "latency"
                                    ] = node_latencies[match][graph.sources[pe][0]]

                    if not found:
                        print("Couldn't find tile port in kernel latencies", kernel)

    return kernel_latencies, stencil_valid_adjust


def update_kernel_latencies(
    dir_name,
    graph,
    id_to_name,
    placement,
    routing,
    existing_kernel_latencies,
    harden_flush,
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    if sparse:
        return

    port_remap = json.load(open(f"{dir_name}/design.port_remap"))

    kernel_latencies, node_latencies = branch_delay_match_within_kernels(
        graph, id_to_name, placement, routing, existing_kernel_latencies, port_remap
    )

    kernel_graph = construct_kernel_graph(graph, kernel_latencies)

    branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing)

    # branch_delay_match_all_nodes(graph, id_to_name, placement, routing)

    flush_latencies, max_flush_cycles = flush_cycles(
        graph, id_to_name, harden_flush, pipeline_config_interval, pes_with_packed_ponds
    )
    for node in kernel_graph.nodes:
        if "io16in" in node.kernel or "io1in" in node.kernel:
            node.latency -= max_flush_cycles
            assert (
                node.latency >= 0
            ), f"{node.kernel} has negative compute kernel latency"

    matched_kernel_latencies, stencil_valid_adjust = calculate_latencies(
        graph,
        kernel_graph,
        node_latencies,
        existing_kernel_latencies,
        port_remap,
        instance_to_instr,
    )
    # updated_kernel_latencies.json only for residual add and manual placed resnet for now
    if os.path.exists(f"{dir_name}/updated_kernel_latencies.json"):
        updated_kernel_latencies = json.load(open(f"{dir_name}/updated_kernel_latencies.json"))
        for kernel, latency_dict in matched_kernel_latencies.items():
            if "hcompute_output_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    if "input_cgra_stencil" or "in2_output_cgra_stencil" in kernel_port:
                        d1["latency"] = updated_kernel_latencies[kernel][kernel_port]["latency"]
            if "_glb_" in kernel:
                matched_kernel_latencies[kernel] = updated_kernel_latencies[kernel]
    # ub_latency.json only for manual placed resnet
    if os.path.exists(f"{dir_name}/ub_latency.json"):
        ub_latencies = json.load(open(f"{dir_name}/ub_latency.json"))
        for kernel, latency_dict in matched_kernel_latencies.items():
            if "hcompute_input_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    port_num = kernel_port.split("_")[-1]
                    d1["latency"] = ub_latencies["input_cgra_stencil"][port_num]["latency"]
            if "hcompute_kernel_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    d1["latency"] = min(value["latency"] for value in ub_latencies["kernel_cgra_stencil"].values())
    matched_flush_latencies = {
        id_to_name[str(mem_id)]: latency for mem_id, latency in flush_latencies.items()
    }

    pond_latencies = {}
    for pond_node in graph.get_ponds():
        for port, lat in pond_node.input_port_latencies.items():
            if port != "flush":
                pond_latencies[id_to_name[pond_node.tile_id]] = lat

    kernel_latencies_file = glob.glob(f"{dir_name}/*_compute_kernel_latencies.json")[0]

    flush_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "flush_latencies"
    )
    pond_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "pond_latencies"
    )
    stencil_valid_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "stencil_valid_latencies"
    )

    fout = open(kernel_latencies_file, "w")
    fout.write(json.dumps(matched_kernel_latencies, indent=4))

    fout = open(flush_latencies_file, "w")
    fout.write(json.dumps(matched_flush_latencies, indent=4))

    fout = open(pond_latencies_file, "w")
    fout.write(json.dumps(pond_latencies, indent=4))

    fout = open(stencil_valid_latencies_file, "w")
    fout.write(json.dumps(stencil_valid_adjust, indent=4))


def segment_node_to_string(node):
    if node[0] == "SB":
        return f"{node[0]} ({node[1]}, {node[2]}, {node[3]}, {node[4]}, {node[5]}, {node[6]})"
    elif node[0] == "PORT":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]})"
    elif node[0] == "REG":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]}, {node[5]})"
    elif node[0] == "RMUX":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]})"


def dump_routing_result(dir_name, routing):
    route_name = os.path.join(dir_name, "design.route")
    fout = open(route_name, "w")

    for net_id, route in routing.items():
        fout.write(f"Net ID: {net_id} Segment Size: {len(route)}\n")
        src = route[0]
        for seg_index, segment in enumerate(route):
            fout.write(f"Segment: {seg_index} Size: {len(segment)}\n")

            for node in segment:
                fout.write(f"{segment_node_to_string(node)}\n")
        fout.write("\n")


def dump_placement_result(dir_name, placement, id_to_name):
    place_name = os.path.join(dir_name, "design.place")
    fout = open(place_name, "w")
    fout.write("Block Name			X	Y		#Block ID\n")
    fout.write("---------------------------\n")

    for tile_id, place in placement.items():
        fout.write(f"{id_to_name[tile_id]}\t\t{place[0]}\t{place[1]}\t\t#{tile_id}\n")


def dump_id_to_name(app_dir, id_to_name):
    id_name = os.path.join(app_dir, "design.id_to_name")
    fout = open(id_name, "w")
    for id_, name in id_to_name.items():
        fout.write(f"{id_}: {name}\n")


def load_id_to_name(id_filename):
    fin = open(id_filename, "r")
    lines = fin.readlines()
    id_to_name = {}

    for line in lines:
        id_to_name[line.split(": ")[0]] = line.split(": ")[1].rstrip()

    return id_to_name

 
def count_primitives(graph):
    counts = defaultdict(int)

    # --- Tiles ---
    for tile in graph.get_tiles():
        t = tile.tile_type
        if t == TileType.PE:
            # pipeline inputs
            for port, latency in tile.input_port_latencies.items():
                # if latency >= 0, port is used and pipelined
                counts["pe_input_reg"] += 1
            # one compute op per cycle
            counts["pe_compute_op"] += 1
        elif t == TileType.MEM:
            # pipeline inputs
            for port, latency in tile.input_port_latencies.items():
                # if latency >= 0, port is used and pipelined
                counts["mem_input_reg"] += 1
            # one compute op per cycle
            counts["mem_compute_op"] += 1

        elif t.name.startswith("IO"):
            counts["io_tile_access"] += 1
        elif t.name.startswith("REG"):
            counts["pipeline_reg"] += 1

    # --- Routes / Switchboxes ---
    for route in graph.get_routes():
        # count total hops
        counts["interconnect_hop"] += 1

    return counts

def count_ic_streams(graph):
    count = 0
    for route in graph.get_routes(): # treat as "tile"
        r = route.route_type
        if r == RouteType.PORT:
            count += 1
    return count

def count_num_pe_tiles(graph):
    count = 0
    for tile in graph.get_tiles():
        t = tile.tile_type
        if t == TileType.PE:
            count += 1
    return count

def count_pe_ports_used(graph):
    pe_ports_used = 0
    for tile in graph.get_tiles():
        t = tile.tile_type
        if t == TileType.PE:
            # pipeline inputs
            #pe_ports_used = 0
            for port, latency in tile.input_port_latencies.items():
                # if latency >= 0, port is used and pipelined
                pe_ports_used += 1
    return pe_ports_used

def count_num_mem_tiles(graph):
    count = 0
    for tile in graph.get_tiles():
        t = tile.tile_type
        if t == TileType.MEM:
            count += 1
    return count

def count_mem_ports_used(graph):
    mem_ports_used = 0
    for tile in graph.get_tiles():
        t = tile.tile_type
        if t == TileType.MEM:
            #mem_ports_used = 0
            # pipeline inputs
            for port, latency in tile.input_port_latencies.items():
                # if latency >= 0, port is used and pipelined
                #counts["mem_input_reg"] += 1
                mem_ports_used += 1
    return mem_ports_used

def num_pe_tiles(graph):
    return sum(1 for tile in graph.get_tiles() if tile.tile_type == TileType.PE)

def num_pe_ports(graph):
    count = 0
    for tile in graph.get_tiles():
        if tile.tile_type == TileType.PE:
            count += len(tile.input_port_latencies)
    return count

def num_mem_tiles(graph):
    return sum(1 for tile in graph.get_tiles() if tile.tile_type == TileType.MEM)

def num_mem_ports(graph):
    count = 0
    for tile in graph.get_tiles():
        if tile.tile_type == TileType.MEM:
            count += len(tile.input_port_latencies)
    return count

def num_io_tiles(graph):
    return sum(1 for tile in graph.get_tiles() if tile.tile_type.name.startswith("IO"))

def num_ic_rmux(graph):
    count = 0
    for route in graph.get_routes():
        if route.route_type == RouteType.RMUX:
            count += 1
    return count

def num_ic_reg(graph):
    count = 0
    for route in graph.get_routes():
        if route.route_type == RouteType.REG:
            count += 1
    return count

def num_ic_ports(graph):
    count = 0
    for route in graph.get_routes():
        if route.route_type == RouteType.PORT:
            count += 1
    return count

def num_ic_sb(graph):
    count = 0
    for route in graph.get_routes():
        if route.route_type == RouteType.SB:
            count += 1
    return count

def num_pipeline_regs(graph):
    count = 0
    for tile in graph.get_tiles():
        if tile.tile_type == TileType.PE or tile.tile_type == TileType.MEM:
            for port, latency in tile.input_port_latencies.items():
                if latency >= 0:
                    count += 1
    return count

# -------------------------------------------------------------------------
# Capstone power model and controller helpers
# -------------------------------------------------------------------------

EVENT_FEATURE_ORDER = [
    "num_pe_tiles",
    "num_pe_ports",
    "num_mem_tiles",
    "num_mem_ports",
    "num_io_tiles",
    "num_ic_rmux",
    "num_ic_reg",
    "num_ic_port",
    "num_ic_sb",
    "num_pipeline_regs",
    "bias",
]

DEFAULT_BETA_DYN = {
    "num_pe_tiles": 0.0,
    "num_pe_ports": 2.250946660015402,
    "num_mem_tiles": 0.9975711826014549,
    "num_mem_ports": 0.7512165803849641,
    "num_io_tiles": 1.6283201695194016,
    "num_ic_rmux": 0.04083986649560815,
    "num_ic_reg": 0.02863778064778701,
    "num_ic_port": 0.0014002861223174122,
    "num_ic_sb": 0.0408846502557773,
    "num_pipeline_regs": 0.011740420157001132,
    "bias": 18.485259876871492,
}

DEFAULT_LEAK_OBJ = {
    "model": "Pleak = sum_j theta[j]*Z[j]  (theta >= 0 ridge, then clamped)",
    "feature_order": [
        "num_pe_tiles",
        "num_mem_tiles",
        "num_io_tiles",
        "num_pipeline_regs",
        "bias",
    ],
    "theta_mW_per_count": {
        "num_pe_tiles": 0.00015618331667396941,
        "num_mem_tiles": 0.0005963323850723349,
        "num_io_tiles": 0.0005963323135265115,
        "num_pipeline_regs": 2.4821606866183925e-05,
        "bias": 0.0,
    },
}

DEFAULT_GAMMA_FIT = {
    "model": "log(gamma) = a*log(proxy) + b",
    "a": 0.3067233202367256,
    "b": -2.361213550819693,
    "proxy_weights": {
        "num_ic_reg": 1.0,
        "num_pipeline_regs": 1.0,
        "num_io_tiles": 0.1,
        "num_ic_rmux": 0.5,
        "num_ic_sb": 0.5,
        "num_ic_port": 0.25,
        "num_pe_tiles": 0.1,
        "num_mem_tiles": 0.2,
        "num_pe_ports": 0.05,
        "num_mem_ports": 0.05,
    },
    "proxy_stats": {
        "proxy_min": 956.1,
        "proxy_med": 2122.375,
        "proxy_max": 2935.2999999999997,
    },
}


def _read_json_if_exists(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"[capstone] Warning: failed to load {path}: {e}. Using default.")
    return copy.deepcopy(default)


def extract_power_features(graph):
    """Compiler-visible event-count vector X(x)."""
    return {
        "num_pe_tiles": float(num_pe_tiles(graph)),
        "num_pe_ports": float(num_pe_ports(graph)),
        "num_mem_tiles": float(num_mem_tiles(graph)),
        "num_mem_ports": float(num_mem_ports(graph)),
        "num_io_tiles": float(num_io_tiles(graph)),
        "num_ic_rmux": float(num_ic_rmux(graph)),
        "num_ic_reg": float(num_ic_reg(graph)),
        "num_ic_port": float(num_ic_ports(graph)),
        "num_ic_sb": float(num_ic_sb(graph)),
        "num_pipeline_regs": float(num_pipeline_regs(graph)),
        "bias": 1.0,
    }


def feat_vec(graph, f_mhz):
    """Feature vector used for candidate diversity/pruning."""
    feats = extract_power_features(graph)
    return tuple(feats.get(k, 0.0) for k in EVENT_FEATURE_ORDER) + (float(round(f_mhz)),)


def cosine(a, b):
    ax = sum(v * v for v in a) ** 0.5
    bx = sum(v * v for v in b) ** 0.5
    if ax == 0 or bx == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (ax * bx)


def load_power_model(scales=None):
    """
    Loads the learned Capstone mean model.

    Expected files, if available:
      coeffs/capstone_dynamic_coeffs.json
      coeffs/leakage_coeffs.json
      coeffs/gamma_proxy_fit.json

    Falls back to hard-coded coefficients.
    """
    scales = scales or {}
    model_dir = Path(scales.get("model_dir", os.environ.get("CAPSTONE_MODEL_DIR", ".")))
    coeff_dir = model_dir / "coeffs"

    beta_dyn = _read_json_if_exists(coeff_dir / "capstone_dynamic_coeffs.json", DEFAULT_BETA_DYN)
    leak_obj = _read_json_if_exists(coeff_dir / "leakage_coeffs.json", DEFAULT_LEAK_OBJ)
    gfit = _read_json_if_exists(coeff_dir / "gamma_proxy_fit.json", DEFAULT_GAMMA_FIT)

    if isinstance(beta_dyn, dict) and "beta_mW_per_count" in beta_dyn:
        beta_dyn = beta_dyn["beta_mW_per_count"]

    theta_leak_map = leak_obj.get("theta_mW_per_count", leak_obj)
    return {
        "beta_dyn": {k: float(v) for k, v in beta_dyn.items()},
        "theta_leak": {k: float(v) for k, v in theta_leak_map.items()},
        "gamma_fit": gfit,
    }


@dataclass
class PowerPrediction:
    total_mW: float
    dyn_mW: float
    leak_mW: float
    gamma_hat: float
    gamma_eff: float
    proxy: float
    feature_counts: Dict[str, float]
    beta_dyn: Dict[str, float]
    theta_leak: Dict[str, float]


def predict_power_components(graph, f_mhz, instance_to_instr, II, scales=None):
    """
    Predict total power and expose the event-count vector needed by Capstone III.

    Dynamic model:
        P_dyn = gamma_eff * sum_e beta_e X_e

    Leakage model:
        P_leak = sum_j theta_j Z_j

    If use_freq_ii_scaling is enabled, gamma_eff additionally scales by
    (f/II)/(f_ref/II_ref).
    """
    scales = scales or {}
    feats = extract_power_features(graph)
    model = load_power_model(scales)
    beta_dyn = model["beta_dyn"]
    theta_leak_map = model["theta_leak"]
    gfit = model["gamma_fit"]

    a = float(gfit.get("a", 0.0))
    b = float(gfit.get("b", 0.0))
    proxy_weights = {k: float(v) for k, v in gfit.get("proxy_weights", {}).items()}

    eps = 1e-12
    proxy = 0.0
    for k, w in proxy_weights.items():
        proxy += w * float(feats.get(k, 0.0))
    proxy = max(proxy, eps)

    gamma_hat = math.exp(a * math.log(proxy) + b)
    gamma_eff = gamma_hat

    if bool(scales.get("use_freq_ii_scaling", False)):
        f_ref = float(scales.get("freq_ref_mhz", 1.0))
        if f_ref <= 0.0:
            raise ValueError("Power-model frequency reference must be positive.")
        II_eff = effective_power_ii(II)
        II_ref = effective_power_ii(scales.get("II_ref", 1.0))
        throughput_scale = (float(f_mhz) / II_eff) / (f_ref / II_ref)
        gamma_eff *= throughput_scale

    dyn_base = 0.0
    for feat, coeff in beta_dyn.items():
        dyn_base += float(coeff) * float(feats.get(feat, 0.0))
    dyn_mW = gamma_eff * dyn_base

    leak_mW = 0.0
    for feat, coeff in theta_leak_map.items():
        leak_mW += float(coeff) * float(feats.get(feat, 0.0))

    total_mW = dyn_mW + leak_mW

    return PowerPrediction(
        total_mW=float(total_mW),
        dyn_mW=float(dyn_mW),
        leak_mW=float(leak_mW),
        gamma_hat=float(gamma_hat),
        gamma_eff=float(gamma_eff),
        proxy=float(proxy),
        feature_counts=feats,
        beta_dyn=beta_dyn,
        theta_leak=theta_leak_map,
    )


def predict_power(graph, f_mhz, instance_to_instr, II, scales=None):
    pred = predict_power_components(graph, f_mhz, instance_to_instr, II, scales=scales)
    print(f"~ ~ Power prediction: ~ ~")
    print(f"  total={pred.total_mW:.2f} mW,")
    print(f"  dynamic={pred.dyn_mW:.2f} mW,")
    print(f"  leakage={pred.leak_mW:.2f} mW,")
    print(f"  gamma_hat={pred.gamma_hat:.3f},")
    print(f"  gamma_eff={pred.gamma_eff:.3f},")
    print(f"  proxy={pred.proxy:.3f},")
    print(f"  features_counts={pred.feature_counts}")
    return pred.total_mW
def _append_csv_row(path, row):
    """
    Append a CSV row while allowing newly added columns.
    This avoids silent header/column mismatch if a trace generated before a
    new column existed is extended with newer rows.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0

    if not exists:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)

    new_fields = old_fields + [k for k in row.keys() if k not in old_fields]
    if new_fields != old_fields:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=new_fields, extrasaction="ignore")
            writer.writeheader()
            for old in old_rows:
                writer.writerow(old)
            writer.writerow(row)
        return

    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_fields, extrasaction="ignore")
        writer.writerow(row)

# ---- Upper-envelope builders ----
@dataclass
class GuardbandSpec:  # Capstone I
    gamma: float  # e.g., 0.15 means 15% margin


def make_guardband_upper(mean_predict):
    def U(graph, f_mhz, II, group, spec: GuardbandSpec):
        mu = float(mean_predict(graph, f_mhz, II))
        return (1.0 + spec.gamma) * mu
    return U


@dataclass
class ConformalSpec:  # Capstone II
    residual_q_mW: Dict[str, float]
    alpha: float
    scale_by_freq: bool = True
    freq_ref_mhz: float = 100.0


def make_conformal_upper(mean_predict):
    def U(graph, f_mhz, II, group, spec: ConformalSpec):
        mu = float(mean_predict(graph, f_mhz, II))
        q = spec.residual_q_mW.get(group, spec.residual_q_mW.get("global", 0.0))
        if spec.freq_ref_mhz <= 0:
            raise ValueError("Conformal frequency reference must be positive.")
        scale = (
            max(1.0, f_mhz / spec.freq_ref_mhz)
            if spec.scale_by_freq
            else 1.0
        )
        return mu + q * scale
    return U


def make_power_model_scales(model_dir, pipeline_config_interval):
    """
    Configure the operating-point scaling required by the power model.

    The released coefficients are referenced to CAPSTONE_FREQ_REF_MHZ, which is
    100 MHz by default. Using the current pipeline II as II_ref makes the online
    scaling reduce to f/f_ref when II is unchanged across candidates. AHA's
    nonpositive/default II is mapped to effective II=1. II_ref is not
    independently configurable because it must describe the current workload.
    """

    freq_ref_mhz = float(os.environ.get("CAPSTONE_FREQ_REF_MHZ", "100.0"))
    if freq_ref_mhz <= 0:
        raise ValueError("CAPSTONE_FREQ_REF_MHZ must be positive.")

    raw_pipeline_ii = float(pipeline_config_interval)
    ii_ref = effective_power_ii(raw_pipeline_ii)

    return {
        "model_dir": model_dir,
        "use_freq_ii_scaling": (
            os.environ.get("CAPSTONE_USE_FREQ_II_SCALING", "1") != "0"
        ),
        "freq_ref_mhz": freq_ref_mhz,
        "II_ref": ii_ref,
        "raw_pipeline_config_interval": raw_pipeline_ii,
    }


def finite_sample_conformal_quantile(normalized_residuals, alpha):
    """
    Return (q, k) using the split-conformal finite-sample index.

    q is None when k > n. For standard deterministic split conformal, no finite
    quantile supports that alpha with the supplied calibration-set size.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"Conformal alpha must lie in (0, 1), got {alpha}.")
    residuals = sorted(float(value) for value in normalized_residuals)
    if any(value < 0.0 or not math.isfinite(value) for value in residuals):
        raise ValueError("Conformal residuals must be finite and nonnegative.")

    n = len(residuals)
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if n == 0 or k > n:
        return None, k
    return residuals[k - 1], k


def _sample_number(sample, names, sample_index):
    for name in names:
        if name in sample and sample[name] not in (None, ""):
            return float(sample[name])
    raise ValueError(
        f"Calibration sample {sample_index} is missing one of: "
        + ", ".join(names)
    )


def load_global_normalized_residuals(
    calibration_json,
    freq_ref_mhz,
    scale_by_freq=True,
):
    """
    Load Capstone II global calibration samples and compute r_i.

    Accepted JSON shapes are a list of sample objects or
    {"samples": [...], "f_ref_mhz": 100.0}. Each sample must provide reference
    power, predicted power, and frequency. Common field-name variants are
    accepted to simplify private calibration-data export.
    """

    path = Path(calibration_json)
    if not path.exists():
        raise FileNotFoundError(f"Capstone II calibration file not found: {path}")
    obj = json.loads(path.read_text())

    if isinstance(obj, list):
        samples = obj
    elif isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        samples = obj["samples"]
        if "f_ref_mhz" in obj:
            freq_ref_mhz = float(obj["f_ref_mhz"])
    else:
        raise ValueError(
            "CAPSTONE_II_CALIBRATION_JSON must contain a list of samples or "
            "an object with a 'samples' list."
        )

    if freq_ref_mhz <= 0:
        raise ValueError("Calibration f_ref_mhz must be positive.")

    assume_reference_frequency = (
        os.environ.get(
            "CAPSTONE_II_ASSUME_F_REF_FOR_MISSING_FREQ",
            "0",
        )
        == "1"
    )
    residuals = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Calibration sample {index} is not an object.")
        p_ref = _sample_number(
            sample,
            (
                "reference_power_mW",
                "ref_power_mW",
                "true_total_mW",
                "P_ref_mW",
            ),
            index,
        )
        mu = _sample_number(
            sample,
            (
                "predicted_power_mW",
                "pred_power_mW",
                "pred_total_mW",
                "P_pred_mW",
            ),
            index,
        )
        try:
            f_mhz = _sample_number(
                sample,
                ("frequency_mhz", "f_mhz", "freq_mhz"),
                index,
            )
        except ValueError:
            if not assume_reference_frequency:
                raise
            f_mhz = freq_ref_mhz

        rho = max(1.0, f_mhz / freq_ref_mhz) if scale_by_freq else 1.0
        residuals.append(max(0.0, p_ref - mu) / rho)

    return residuals, float(freq_ref_mhz)


def configure_capstone_ii_conformal_specs():
    """
    Build global Capstone II anchor/speculative envelopes.

    With n=24 and alpha_anchor=0.005, the anchor quantile is not finite. The
    default 'disable' policy sets q_anchor=+inf and therefore returns no claimed
    anchor while allowing every other controller to continue. Supported anchor
    fallback policies are:
      - disable: no finite anchor and no guarantee claim
      - engineering: require CAPSTONE_II_ANCHOR_FALLBACK_Q_MW
      - empirical_max: use max residual without a 99.5% conformal claim
      - error: stop immediately
    Explicit CAPSTONE_II_ANCHOR_Q_MW or CAPSTONE_II_SPEC_Q_MW values override
    calibrated values and are recorded as explicit engineering inputs.
    """

    alpha_anchor = float(
        os.environ.get("CAPSTONE_II_ANCHOR_ALPHA", "0.005")
    )
    alpha_spec = float(os.environ.get("CAPSTONE_II_SPEC_ALPHA", "0.05"))
    scale_by_freq = os.environ.get("CAPSTONE_II_SCALE_BY_FREQ", "1") != "0"
    freq_ref_mhz = float(os.environ.get("CAPSTONE_FREQ_REF_MHZ", "100.0"))
    calibration_path = os.environ.get(
        "CAPSTONE_II_CALIBRATION_JSON",
        "",
    ).strip()

    residuals = []
    if calibration_path:
        residuals, freq_ref_mhz = load_global_normalized_residuals(
            calibration_path,
            freq_ref_mhz,
            scale_by_freq=scale_by_freq,
        )

    q_anchor_cal, k_anchor = finite_sample_conformal_quantile(
        residuals,
        alpha_anchor,
    )
    q_spec_cal, k_spec = finite_sample_conformal_quantile(
        residuals,
        alpha_spec,
    )
    n_global = len(residuals)

    explicit_anchor = os.environ.get("CAPSTONE_II_ANCHOR_Q_MW", "").strip()
    explicit_spec = os.environ.get("CAPSTONE_II_SPEC_Q_MW", "").strip()

    if explicit_spec:
        q_spec = float(explicit_spec)
        spec_source = "explicit engineering input"
    elif q_spec_cal is not None:
        q_spec = q_spec_cal
        spec_source = "finite-sample global conformal quantile"
    else:
        q_spec = float("inf")
        spec_source = "unavailable: insufficient global calibration samples"

    anchor_has_finite_conformal_guarantee = q_anchor_cal is not None
    if explicit_anchor:
        q_anchor = float(explicit_anchor)
        anchor_source = "explicit engineering input; no conformal claim"
        anchor_has_finite_conformal_guarantee = False
    elif q_anchor_cal is not None:
        q_anchor = q_anchor_cal
        anchor_source = "finite-sample global conformal quantile"
    else:
        fallback_policy = os.environ.get(
            "CAPSTONE_II_ANCHOR_FALLBACK_POLICY",
            "disable",
        ).strip().lower()
        if fallback_policy == "disable":
            q_anchor = float("inf")
            anchor_source = (
                "disabled: n_global is insufficient for the requested alpha"
            )
        elif fallback_policy == "engineering":
            fallback_q = os.environ.get(
                "CAPSTONE_II_ANCHOR_FALLBACK_Q_MW",
                "",
            ).strip()
            if not fallback_q:
                raise ValueError(
                    "CAPSTONE_II_ANCHOR_FALLBACK_POLICY=engineering requires "
                    "CAPSTONE_II_ANCHOR_FALLBACK_Q_MW."
                )
            q_anchor = float(fallback_q)
            anchor_source = "engineering fallback; no conformal claim"
        elif fallback_policy == "empirical_max":
            if not residuals:
                raise ValueError(
                    "The empirical_max fallback requires calibration samples."
                )
            q_anchor = max(residuals)
            anchor_source = (
                "empirical maximum fallback; no 99.5% conformal claim"
            )
        elif fallback_policy == "error":
            minimum_n = int(
                math.ceil((1.0 - alpha_anchor) / alpha_anchor)
            )
            raise ValueError(
                f"n_global={n_global} cannot support alpha_anchor="
                f"{alpha_anchor}. At least {minimum_n} samples are required."
            )
        else:
            raise ValueError(
                "Unknown CAPSTONE_II_ANCHOR_FALLBACK_POLICY="
                f"{fallback_policy}."
            )
        anchor_has_finite_conformal_guarantee = False

    if q_anchor < 0.0 or q_spec < 0.0:
        raise ValueError("Capstone II residual quantiles must be nonnegative.")
    if math.isfinite(q_anchor) and math.isfinite(q_spec) and q_anchor < q_spec:
        raise ValueError(
            "Capstone II requires q_anchor >= q_spec because "
            "alpha_anchor < alpha_spec."
        )

    anchor_spec = ConformalSpec(
        residual_q_mW={"global": float(q_anchor)},
        alpha=alpha_anchor,
        scale_by_freq=scale_by_freq,
        freq_ref_mhz=freq_ref_mhz,
    )
    spec_spec = ConformalSpec(
        residual_q_mW={"global": float(q_spec)},
        alpha=alpha_spec,
        scale_by_freq=scale_by_freq,
        freq_ref_mhz=freq_ref_mhz,
    )
    metadata = {
        "calibration_json": calibration_path or None,
        "n_global": n_global,
        "anchor_k": k_anchor,
        "spec_k": k_spec,
        "anchor_q_mW": None if not math.isfinite(q_anchor) else q_anchor,
        "spec_q_mW": None if not math.isfinite(q_spec) else q_spec,
        "anchor_alpha": alpha_anchor,
        "spec_alpha": alpha_spec,
        "anchor_source": anchor_source,
        "spec_source": spec_source,
        "anchor_has_finite_conformal_guarantee": (
            anchor_has_finite_conformal_guarantee
        ),
        "scale_by_freq": scale_by_freq,
        "freq_ref_mhz": freq_ref_mhz,
    }
    return anchor_spec, spec_spec, metadata


@dataclass
class RobustBoundSpec:  # Capstone III
    """
    Event-level bounded-error model.

    Capstone III bound is:
        U_rob(x) = sum_e (beta_e + epsilon_e) X_e(x) + P_leak + epsilon_leak

    In this implementation, beta_e is the pre-gamma dynamic coefficient used by
    predict_power_components, so event epsilons are also interpreted in the same
    pre-gamma coefficient units and are multiplied by gamma_eff.
    """
    eps_fit_mW_per_count: Dict[str, float]
    eps_act_mW_per_count: Dict[str, float]
    eps_pvt_mW_per_count: Dict[str, float]
    eps_ood_mW_per_count: Dict[str, float]
    eps_leak_mW: float = 0.0
    use_budgeted_uncertainty: bool = False
    gamma_budget: Optional[float] = None

    def event_eps_total(self):
        keys = set(EVENT_FEATURE_ORDER)
        keys |= set(self.eps_fit_mW_per_count)
        keys |= set(self.eps_act_mW_per_count)
        keys |= set(self.eps_pvt_mW_per_count)
        keys |= set(self.eps_ood_mW_per_count)
        return {
            k: float(self.eps_fit_mW_per_count.get(k, 0.0))
             + float(self.eps_act_mW_per_count.get(k, 0.0))
             + float(self.eps_pvt_mW_per_count.get(k, 0.0))
             + float(self.eps_ood_mW_per_count.get(k, 0.0))
            for k in keys
        }


def _coerce_bound_map(obj, key, default):
    """Accepts either *_mW_per_count or shorter legacy JSON keys."""
    if key in obj and isinstance(obj[key], dict):
        return {k: float(v) for k, v in obj[key].items()}
    short = key.replace("_mW_per_count", "")
    if short in obj and isinstance(obj[short], dict):
        return {k: float(v) for k, v in obj[short].items()}
    return copy.deepcopy(default)


def load_robust_bound_spec(model_dir="."):
    """
    Loads Capstone III event-level bounds.

    Preferred JSON format:
    {
      "eps_fit_mW_per_count": {"num_ic_reg": 0.001, ...},
      "eps_act_mW_per_count":  {...},
      "eps_pvt_mW_per_count":  {...},
      "eps_ood_mW_per_count":  {...},
      "eps_leak_mW": 0.0,
      "use_budgeted_uncertainty": false,
      "gamma_budget": null
    }

    File search order:
      1. CAPSTONE_BOUNDS_JSON, if set
      2. <model_dir>/coeffs/capstone_iii_bounds.json

    If no file exists, uses a conservative relative epsilon controlled by
    CAPSTONE_III_DEFAULT_REL_EPS. When set to 0.0, the bounded mode reduces to
    the mean model.
    """
    model_dir = Path(model_dir)
    env_path = os.environ.get("CAPSTONE_BOUNDS_JSON", "").strip()
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.append(model_dir / "coeffs" / "capstone_iii_bounds.json")

    obj = None
    for path in candidate_paths:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
                print(f"[bounded] Loaded Capstone III bounds from {path}")
                break
            except Exception as e:
                print(f"[bounded] Warning: failed to read {path}: {e}")

    model = load_power_model({"model_dir": str(model_dir)})
    beta_dyn = model["beta_dyn"]
    default_rel = float(os.environ.get("CAPSTONE_III_DEFAULT_REL_EPS", "0.15"))
    default_eps = {k: abs(float(v)) * default_rel for k, v in beta_dyn.items()}

    if obj is None:
        print(f"[bounded] No bounds JSON found; using {default_rel:.3f} relative event bounds.")
        return RobustBoundSpec(
            eps_fit_mW_per_count=default_eps,
            eps_act_mW_per_count={},
            eps_pvt_mW_per_count={},
            eps_ood_mW_per_count={},
            eps_leak_mW=float(os.environ.get("CAPSTONE_III_EPS_LEAK_MW", "0.0")),
        )

    return RobustBoundSpec(
        eps_fit_mW_per_count=_coerce_bound_map(obj, "eps_fit_mW_per_count", default_eps),
        eps_act_mW_per_count=_coerce_bound_map(obj, "eps_act_mW_per_count", {}),
        eps_pvt_mW_per_count=_coerce_bound_map(obj, "eps_pvt_mW_per_count", {}),
        eps_ood_mW_per_count=_coerce_bound_map(obj, "eps_ood_mW_per_count", {}),
        eps_leak_mW=float(obj.get("eps_leak_mW", obj.get("eps_leak", 0.0))),
        use_budgeted_uncertainty=bool(obj.get("use_budgeted_uncertainty", False)),
        gamma_budget=obj.get("gamma_budget", None),
    )


def robust_upper_bound(pred: PowerPrediction, spec: RobustBoundSpec):
    """Capstone III robust upper envelope."""
    feats = pred.feature_counts
    beta = pred.beta_dyn
    eps_total = spec.event_eps_total()

    base = 0.0
    increments = []
    for e, coeff in beta.items():
        x_e = float(feats.get(e, 0.0))
        base += float(coeff) * x_e
        inc = max(0.0, float(eps_total.get(e, 0.0)) * x_e)
        increments.append(inc)

    if spec.use_budgeted_uncertainty:
        gamma_budget = spec.gamma_budget
        if gamma_budget is None:
            gamma_budget = len(increments)
        gamma_budget = max(0.0, float(gamma_budget))
        increments.sort(reverse=True)
        whole = int(math.floor(gamma_budget))
        frac = gamma_budget - whole
        err = sum(increments[:whole])
        if whole < len(increments):
            err += frac * increments[whole]
        dyn_upper = pred.gamma_eff * (base + err)
    else:
        dyn_upper = pred.gamma_eff * (base + sum(increments))

    return float(dyn_upper + pred.leak_mW + float(spec.eps_leak_mW))


@dataclass
class OnlinePlanConfig:
    cap_mW: float
    K_outputs: int = 4
    stop_on: str = "spec" # 'spec' or 'anchor'
    diversity_weight: float = 0.25
    min_delta_freq_MHz: float = 1.0


@dataclass
class CandidateSnap:
    # Number of critical-path breaks already applied to produce this candidate.
    itr: int
    f_mhz: float
    power_mean_mW: float
    power_upper_mW: float
    feat: tuple
    robust_slack_mW: Optional[float] = None
    tag: str = ""
    # Search loop iteration that evaluated this candidate. It currently matches
    # itr because each successful iteration inserts one critical-path break,
    # but is tracked separately so the log remains correct if that changes.
    iteration: Optional[int] = None


class OnlinePlanner:
    """
    Used by Capstone I/II. Keeps one safe anchor plus up to K-1 speculative
    candidates while pipelining increases frequency.
    """
    def __init__(self, cfg: OnlinePlanConfig):
        self.cfg = cfg
        self.anchor: Optional[CandidateSnap] = None
        self.specs: List[CandidateSnap] = []
        self.last_f = -1.0

    def consider(self, itr: int, graph, f_mhz: float, II: int,
                 group: Optional[str],
                 upper_anchor_fn, anchor_spec,
                 upper_spec_fn, spec_spec,
                 mean_predict, iteration: Optional[int] = None):
        if f_mhz < self.last_f + self.cfg.min_delta_freq_MHz and self.last_f > 0:
            return False

        mu = float(mean_predict(graph, f_mhz, II))
        U_anchor = upper_anchor_fn(graph, f_mhz, II, group, anchor_spec)
        U_spec = upper_spec_fn(graph, f_mhz, II, group, spec_spec)
        feat = feat_vec(graph, f_mhz)
        evaluated_iteration = int(itr if iteration is None else iteration)
        print(
            f"* * * DEBUG: iteration={evaluated_iteration} breaks={itr} "
            f"mu={mu:.3f} U_anchor={U_anchor:.3f} U_spec={U_spec:.3f} "
            f"cap={self.cfg.cap_mW:.3f}"
        )

        if U_anchor <= self.cfg.cap_mW:
            self.anchor = CandidateSnap(itr, f_mhz, mu, U_anchor, feat,
                                        robust_slack_mW=self.cfg.cap_mW - U_anchor,
                                        tag="ANCHOR",
                                        iteration=evaluated_iteration)

        if U_spec <= self.cfg.cap_mW:
            self._maybe_add_spec(CandidateSnap(itr, f_mhz, mu, U_spec, feat,
                                               robust_slack_mW=self.cfg.cap_mW - U_spec,
                                               tag="SPEC",
                                               iteration=evaluated_iteration))

        self.last_f = f_mhz
        stop_key = U_spec if self.cfg.stop_on == "spec" else U_anchor
        return stop_key > self.cfg.cap_mW

    def _maybe_add_spec(self, cand: CandidateSnap):
        def score(c):
            div_pen = 0.0
            for s in self.specs:
                div_pen += cosine(c.feat, s.feat)
            return c.f_mhz - self.cfg.diversity_weight * div_pen

        if len(self.specs) < max(0, self.cfg.K_outputs - 1):
            self.specs.append(cand)
            self.specs.sort(key=lambda c: c.f_mhz, reverse=True)
            return

        worst_idx, worst_sc = None, 1e18
        for idx, s in enumerate(self.specs):
            sc = score(s)
            if sc < worst_sc:
                worst_sc = sc
                worst_idx = idx
        cand_sc = score(cand)
        if cand_sc > worst_sc + 1e-6:
            self.specs[worst_idx] = cand
            self.specs.sort(key=lambda c: c.f_mhz, reverse=True)

    def finalize(self):
        # An anytime-safe output set is valid only if it contains its principal
        # anchor. Speculative candidates must never be returned on their own.
        if self.anchor is None:
            return []

        out = [self.anchor]
        for s in self.specs:
            if len(out) >= self.cfg.K_outputs:
                break
            if s.itr == self.anchor.itr:
                continue
            out.append(s)
        return out


class BoundedPlanner:
    """Capstone III controller."""
    def __init__(self, cfg: OnlinePlanConfig):
        self.cfg = cfg
        self.anchor: Optional[CandidateSnap] = None
        self.safe_candidates: List[CandidateSnap] = []
        self.last_f = -1.0

    def consider(self, cand: CandidateSnap):
        cand.robust_slack_mW = self.cfg.cap_mW - cand.power_upper_mW
        print(f"  [bounded] breaks={cand.itr}, f={cand.f_mhz:.1f} MHz, "
              f"mean={cand.power_mean_mW:.3f} mW, U_rob={cand.power_upper_mW:.3f} mW, "
              f"headroom={cand.robust_slack_mW:.3f} mW")

        if cand.power_upper_mW <= self.cfg.cap_mW + 1e-12:
            cand.tag = "ROBUST_ANCHOR_CANDIDATE"
            self.safe_candidates.append(cand)
            if self.anchor is None or self._better_anchor(cand, self.anchor):
                self.anchor = cand

        self.last_f = cand.f_mhz
        # Monotone stopping is safe for the post-PnR use case when power is
        # assumed to increase with added pipelining. Disable for debugging with
        # CAPSTONE_MONOTONE_STOP=0.
        monotone_stop = os.environ.get("CAPSTONE_MONOTONE_STOP", "1") != "0"
        return monotone_stop and self.anchor is not None and cand.power_upper_mW > self.cfg.cap_mW

    @staticmethod
    def _better_anchor(a: CandidateSnap, b: CandidateSnap):
        # Primary objective: maximize safe frequency.
        # Tie-breaker: minimize remaining robust headroom.
        a_slack = float("inf") if a.robust_slack_mW is None else a.robust_slack_mW
        b_slack = float("inf") if b.robust_slack_mW is None else b.robust_slack_mW
        return (a.f_mhz > b.f_mhz + 1e-9) or (
            abs(a.f_mhz - b.f_mhz) <= 1e-9 and a_slack < b_slack
        )

    def finalize(self):
        if self.anchor is None:
            return []

        others = [c for c in self.safe_candidates if c.itr != self.anchor.itr]

        # Do not Pareto-prune all monotone-safe candidates away.
        # Instead, sort by objectives and let the diversity selector choose up to K-1.
        others.sort(
            key=lambda c: (
                -c.f_mhz,
                c.robust_slack_mW if c.robust_slack_mW is not None else float("inf"),
            )
        )

        others = greedy_diverse_select(
            others,
            k=max(0, self.cfg.K_outputs - 1),
            diversity_weight=self.cfg.diversity_weight,
        )
        self.anchor.tag = "ANCHOR"
        for c in others:
            c.tag = "SPEC"
        return [self.anchor] + others


def pareto_prune_bounded_candidates(candidates: List[CandidateSnap]):
    """
    Remove candidates dominated under the objective:
      - higher frequency is better
      - lower robust headroom is better because it lands closer to the cap
    Feature diversity is handled by the greedy selector after pruning.
    """
    keep = []
    for a in candidates:
        a_slack = float("inf") if a.robust_slack_mW is None else a.robust_slack_mW
        dominated = False
        for b in candidates:
            if a is b:
                continue
            b_slack = float("inf") if b.robust_slack_mW is None else b.robust_slack_mW
            at_least_as_good = (b.f_mhz >= a.f_mhz - 1e-9) and (b_slack <= a_slack + 1e-9)
            strictly_better = (b.f_mhz > a.f_mhz + 1e-9) or (b_slack < a_slack - 1e-9)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    keep.sort(key=lambda c: (-c.f_mhz, c.robust_slack_mW if c.robust_slack_mW is not None else float("inf")))
    return keep


def greedy_diverse_select(candidates: List[CandidateSnap], k: int, diversity_weight: float):
    selected = []
    pool = list(candidates)
    while pool and len(selected) < k:
        def score(c):
            slack = 0.0 if c.robust_slack_mW is None else c.robust_slack_mW
            div_pen = sum(cosine(c.feat, s.feat) for s in selected)
            # Frequency dominates. Slack pushes toward smaller headroom. Diversity avoids duplicates.
            return c.f_mhz - 1e-3 * slack - diversity_weight * div_pen
        best = max(pool, key=score)
        selected.append(best)
        pool.remove(best)
    return selected


# -----------------------------------------------------------------------------
# Simultaneous baseline / Capstone I / Capstone II / Capstone III evaluation
# -----------------------------------------------------------------------------

TABLE8_BOUND_MODE_ORDER = (
    "fit_1x",
    "fit_2x",
    "fit_activity",
    "fit_activity_pvt",
    "full",
)


@dataclass
class SimultaneousModeState:
    """Independent controller state evaluated on a shared candidate stream."""

    key: str
    label: str
    family: str
    planner: Any
    bound_mode: str = ""
    robust_spec: Optional[RobustBoundSpec] = None
    upper_fn: Any = None
    anchor_spec: Any = None
    spec_spec: Any = None
    active: bool = True
    stopped_at_breaks: Optional[int] = None
    stopped_at_iteration: Optional[int] = None
    crossing_upper_mW: Optional[float] = None
    evaluated_candidates: int = 0


def build_table8_bound_specs(full_spec: RobustBoundSpec):
    """
    Derive every Table VIII bound construction from one full-bounds calibration.

    The source JSON should contain the calibrated fit, activity, PVT, and OOD
    maps. The 2x-fit row doubles only epsilon_fit. Leakage uncertainty and the
    budgeted-uncertainty settings are held constant across the five rows.
    """

    fit = copy.deepcopy(full_spec.eps_fit_mW_per_count)
    activity = copy.deepcopy(full_spec.eps_act_mW_per_count)
    pvt = copy.deepcopy(full_spec.eps_pvt_mW_per_count)
    ood = copy.deepcopy(full_spec.eps_ood_mW_per_count)
    fit_2x = {key: 2.0 * float(value) for key, value in fit.items()}

    def make(eps_fit, eps_act=None, eps_pvt=None, eps_ood=None):
        return RobustBoundSpec(
            eps_fit_mW_per_count=copy.deepcopy(eps_fit),
            eps_act_mW_per_count=copy.deepcopy(eps_act or {}),
            eps_pvt_mW_per_count=copy.deepcopy(eps_pvt or {}),
            eps_ood_mW_per_count=copy.deepcopy(eps_ood or {}),
            eps_leak_mW=float(full_spec.eps_leak_mW),
            use_budgeted_uncertainty=bool(full_spec.use_budgeted_uncertainty),
            gamma_budget=full_spec.gamma_budget,
        )

    return {
        "fit_1x": make(fit),
        "fit_2x": make(fit_2x),
        "fit_activity": make(fit, activity),
        "fit_activity_pvt": make(fit, activity, pvt),
        "full": make(fit, activity, pvt, ood),
    }


def select_top_bounded_candidates(candidates: List[CandidateSnap], k: int):
    """
    Retain the top K robust-safe candidates for the Table VIII pruning rows.

    Frequency is the primary objective. Remaining headroom is the tie-breaker.
    This is intentionally separate from BoundedPlanner.finalize(), whose greedy
    diversity policy remains available for normal multi-bitstream generation.
    """

    if k <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.f_mhz,
            candidate.robust_slack_mW
            if candidate.robust_slack_mW is not None
            else float("inf"),
            candidate.itr,
        ),
    )
    return ordered[:k]


def _parse_positive_int_list(raw_value: str, default):
    values = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        if value not in values:
            values.append(value)
    return values or list(default)


def _candidate_to_dict(candidate: Optional[CandidateSnap]):
    if candidate is None:
        return None
    return {
        "iteration": int(
            candidate.itr
            if candidate.iteration is None
            else candidate.iteration
        ),
        "breaks": int(candidate.itr),
        "f_mhz": float(candidate.f_mhz),
        "P_mean_mW": float(candidate.power_mean_mW),
        "P_upper_mW": float(candidate.power_upper_mW),
        "predicted_headroom_mW": (
            None
            if candidate.robust_slack_mW is None
            or not math.isfinite(candidate.robust_slack_mW)
            else float(candidate.robust_slack_mW)
        ),
        "tag": candidate.tag,
    }


def _replay_post_pnr_candidate(
    app_dir,
    break_count,
    placement_save,
    routing_save,
    id_to_name_save,
    netlist,
    pe_cycles,
    io_cycles,
    existing_kernel_latencies,
    harden_flush,
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    """Reconstruct one selected candidate from the unpipelined PnR result."""

    placement = copy.deepcopy(placement_save)
    routing = copy.deepcopy(routing_save)
    id_to_name = copy.deepcopy(id_to_name_save)
    graph = construct_graph(
        placement,
        routing,
        id_to_name,
        netlist,
        pe_latency=pe_cycles,
        pond_latency=0,
        io_latency=io_cycles,
        sparse=sparse,
    )

    replay_latencies = copy.deepcopy(existing_kernel_latencies)
    update_kernel_latencies(
        app_dir,
        graph,
        id_to_name,
        placement,
        routing,
        replay_latencies,
        harden_flush,
        instance_to_instr,
        pipeline_config_interval,
        pes_with_packed_ponds,
        sparse,
    )

    for _ in range(int(break_count)):
        _, crit_path, _ = sta(graph)
        break_crit_path(graph, id_to_name, crit_path, placement, routing)
        graph.regs = None
        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            replay_latencies,
            harden_flush,
            instance_to_instr,
            pipeline_config_interval,
            pes_with_packed_ponds,
            sparse,
        )

    curr_freq, crit_path, crit_nets = sta(graph)
    return (
        graph,
        curr_freq,
        crit_path,
        crit_nets,
        placement,
        routing,
        id_to_name,
    )


def run_all_capstone_modes_post_pnr(
    app_dir,
    graph,
    curr_freq,
    crit_path,
    placement,
    routing,
    id_to_name,
    placement_save,
    routing_save,
    id_to_name_save,
    netlist,
    pe_cycles,
    io_cycles,
    existing_kernel_latencies,
    harden_flush,
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    """
    Evaluate every controller on one post-PnR pipelining trajectory.

    A controller is deactivated when its stopping upper bound crosses the cap.
    The shared graph continues to receive critical-path breaks while any mode is
    active. The infinite-cap baseline therefore continues until maximal
    pipelining, POST_PNR_ITR, or CAPSTONE_MAX_BREAKS ends the trajectory.
    """

    power_cap_mW = float(os.environ.get("CAPSTONE_POWER_CAP_MW", "500.0"))
    k_outputs = int(os.environ.get("NUM_BITSTREAMS", "4"))
    model_dir = os.environ.get("CAPSTONE_MODEL_DIR", ".")
    power_scales = make_power_model_scales(
        model_dir,
        pipeline_config_interval,
    )
    replay_reference_latencies = copy.deepcopy(existing_kernel_latencies)
    kernel = os.path.basename(os.path.abspath(app_dir))
    run_id = os.environ.get(
        "CAPSTONE_RUN_ID",
        f"{kernel}-{int(time.time())}",
    )

    trace_csv = os.environ.get(
        "CAPSTONE_ALL_MODES_TRACE_CSV",
        os.path.join(app_dir, "capstone_all_modes_trace.csv"),
    ).strip()
    summary_csv = os.environ.get(
        "CAPSTONE_ALL_MODES_SUMMARY_CSV",
        os.path.join(app_dir, "capstone_all_modes_summary.csv"),
    ).strip()
    bitstreams_csv = os.environ.get(
        "CAPSTONE_ALL_MODES_BITSTREAMS_CSV",
        os.path.join(app_dir, "capstone_all_modes_bitstreams.csv"),
    ).strip()
    timing_csv = os.environ.get(
        "CAPSTONE_ALL_MODES_TIMING_CSV",
        os.path.join(app_dir, "capstone_figure11_timing.csv"),
    ).strip()
    selection_json = os.environ.get(
        "CAPSTONE_ALL_MODES_SELECTION_JSON",
        os.path.join(app_dir, "capstone_all_modes_selection.json"),
    ).strip()

    # Paper defaults can be overridden without editing this file.
    gb_anchor = GuardbandSpec(
        gamma=float(os.environ.get("CAPSTONE_I_ANCHOR_GAMMA", "0.45"))
    )
    gb_spec = GuardbandSpec(
        gamma=float(os.environ.get("CAPSTONE_I_SPEC_GAMMA", "0.3"))
    )
    (
        conf_anchor,
        conf_spec,
        conformal_metadata,
    ) = configure_capstone_ii_conformal_specs()

    print(
        "[capstone] Mean-power frequency/II scaling: "
        f"enabled={power_scales['use_freq_ii_scaling']}, "
        f"f_ref={power_scales['freq_ref_mhz']:.3f} MHz, "
        f"raw AHA II={power_scales['raw_pipeline_config_interval']:.3f}, "
        f"effective II={power_scales['II_ref']:.3f}"
    )
    print(
        "[capstone] Capstone II global calibration: "
        f"n={conformal_metadata['n_global']}, "
        f"spec={conformal_metadata['spec_source']}, "
        f"anchor={conformal_metadata['anchor_source']}"
    )

    # The same upper-envelope functions are shared because each candidate's mean
    # prediction is cached before any controller is evaluated.
    # The mean power is the raw prediction from the model before the upper bound is applied.
    current_mean = {"value": 0.0}

    def cached_mean_predict(_graph, _f_mhz, _II):
        return current_mean["value"]

    guardband_upper = make_guardband_upper(cached_mean_predict)
    conformal_upper = make_conformal_upper(cached_mean_predict)

    planner_config = {
        "K_outputs": k_outputs,
        "stop_on": "spec",
        "diversity_weight": float(
            os.environ.get("CAPSTONE_DIVERSITY_WEIGHT", "0.01")
        ),
        # Evaluate every generated candidate. The planner still retains the
        # highest-frequency safe anchor.
        "min_delta_freq_MHz": 0.0,
    }

    modes = [
        SimultaneousModeState(
            key="baseline",
            label="Baseline (uncapped guardband)",
            family="guardband",
            planner=OnlinePlanner(
                OnlinePlanConfig(cap_mW=float("inf"), **planner_config)
            ),
            upper_fn=guardband_upper,
            anchor_spec=gb_anchor,
            spec_spec=gb_spec,
        ),
        SimultaneousModeState(
            key="capstone_i",
            label="Capstone I (guardband)",
            family="guardband",
            planner=OnlinePlanner(
                OnlinePlanConfig(cap_mW=power_cap_mW, **planner_config)
            ),
            upper_fn=guardband_upper,
            anchor_spec=gb_anchor,
            spec_spec=gb_spec,
        ),
        SimultaneousModeState(
            key="capstone_ii",
            label="Capstone II (conformal)",
            family="conformal",
            planner=OnlinePlanner(
                OnlinePlanConfig(cap_mW=power_cap_mW, **planner_config)
            ),
            upper_fn=conformal_upper,
            anchor_spec=conf_anchor,
            spec_spec=conf_spec,
        ),
    ]

    bounds_env_path = os.environ.get("CAPSTONE_BOUNDS_JSON", "").strip()
    default_bounds_path = Path(model_dir) / "coeffs" / "capstone_iii_bounds.json"
    require_table8_bounds = (
        os.environ.get(
            "CAPSTONE_REQUIRE_TABLE8_BOUNDS",
            os.environ.get("CAPSTONE_REQUIRE_TABLE7_BOUNDS", "1"),
        ) != "0"
    )
    if (
        require_table8_bounds
        and not (bounds_env_path and Path(bounds_env_path).exists())
        and not default_bounds_path.exists()
    ):
        raise FileNotFoundError(
            "Simultaneous Table VIII evaluation requires a full Capstone III "
            "bounds JSON. Set CAPSTONE_BOUNDS_JSON or place "
            "coeffs/capstone_iii_bounds.json under CAPSTONE_MODEL_DIR. "
        )

    full_bound_spec = load_robust_bound_spec(model_dir=model_dir)
    missing_bound_components = [
        name
        for name, values in (
            ("fit", full_bound_spec.eps_fit_mW_per_count),
            ("activity", full_bound_spec.eps_act_mW_per_count),
            ("PVT", full_bound_spec.eps_pvt_mW_per_count),
            ("OOD", full_bound_spec.eps_ood_mW_per_count),
        )
        if not values
    ]
    if require_table8_bounds and missing_bound_components:
        raise ValueError(
            "The full bounds JSON is missing nonempty Table VIII components: "
            + ", ".join(missing_bound_components)
        )
    table8_specs = build_table8_bound_specs(full_bound_spec)
    for bound_mode in TABLE8_BOUND_MODE_ORDER:
        modes.append(
            SimultaneousModeState(
                key=f"capstone_iii_{bound_mode}",
                label=f"Capstone III ({bound_mode})",
                family="bounded",
                bound_mode=bound_mode,
                planner=BoundedPlanner(
                    OnlinePlanConfig(
                        cap_mW=power_cap_mW,
                        K_outputs=k_outputs,
                        stop_on="anchor",
                        diversity_weight=float(
                            os.environ.get(
                                "CAPSTONE_DIVERSITY_WEIGHT",
                                "0.01",
                            )
                        ),
                        min_delta_freq_MHz=0.0,
                    )
                ),
                robust_spec=table8_specs[bound_mode],
            )
        )

    mode_by_key = {mode.key: mode for mode in modes}
    stop_on_crossing = (
        os.environ.get("CAPSTONE_PER_MODE_MONOTONE_STOP", "1") != "0"
    )
    predictor_total_s = 0.0
    predictor_calls = 0
    sta_total_s = 0.0
    pipelining_total_s = 0.0
    post_pnr_iterations_total_s = 0.0
    successful_iterations = 0
    controller_total_s = {mode.key: 0.0 for mode in modes}
    mode_compile_time_s = {mode.key: 0.0 for mode in modes}

    def log_trace_row(
        mode,
        iteration_count,
        break_count,
        f_mhz,
        pred,
        upper_anchor,
        upper_spec,
        upper_robust,
        stopped_here,
    ):
        if not trace_csv:
            return
        cap = mode.planner.cfg.cap_mW
        effective_upper = (
            upper_robust
            if upper_robust is not None
            else upper_spec
        )
        _append_csv_row(
            trace_csv,
            {
                "run_id": run_id,
                "kernel": kernel,
                "mode": mode.key,
                "bound_mode": mode.bound_mode,
                "iteration": int(iteration_count),
                "breaks": int(break_count),
                "f_mhz": float(f_mhz),
                "II": float(pipeline_config_interval),
                "raw_AHA_II": float(pipeline_config_interval),
                "effective_power_II": float(
                    effective_power_ii(pipeline_config_interval)
                ),
                "P_mean_mW": float(pred.total_mW),
                "P_mean_dyn_mW": float(pred.dyn_mW),
                "P_mean_leak_mW": float(pred.leak_mW),
                "P_upper_anchor_mW": (
                    "" if upper_anchor is None else float(upper_anchor)
                ),
                "P_upper_spec_mW": (
                    "" if upper_spec is None else float(upper_spec)
                ),
                "P_upper_robust_mW": (
                    "" if upper_robust is None else float(upper_robust)
                ),
                "P_effective_stop_upper_mW": float(effective_upper),
                "cap_mW": "" if not math.isfinite(cap) else float(cap),
                "safe": bool(effective_upper <= cap),
                "stopped_here": bool(stopped_here),
                "active_after": bool(mode.active),
                # Merge only NDA-permitted signoff aggregates later.
                "P_oracle_mW": "",
                "success": "",
                "delta_cap_pct": "",
            },
        )

    def evaluate_candidate(
        iteration_count,
        break_count,
        candidate_graph,
        f_mhz,
    ):
        evaluation_start = time.perf_counter()
        predictor_start = time.perf_counter()
        pred = predict_power_components(
            candidate_graph,
            f_mhz,
            instance_to_instr,
            pipeline_config_interval,
            scales=power_scales,
        )
        predictor_elapsed_s = time.perf_counter() - predictor_start
        current_mean["value"] = pred.total_mW

        print(
            f"\n[all-modes] candidate iteration={iteration_count}, "
            f"breaks={break_count}, "
            f"f={f_mhz:.3f} MHz, mean={pred.total_mW:.3f} mW"
        )

        controller_elapsed_s = {}
        for mode in modes:
            if not mode.active:
                continue

            controller_start = time.perf_counter()
            mode.evaluated_candidates += 1
            upper_anchor = None
            upper_spec = None
            upper_robust = None

            if mode.family in {"guardband", "conformal"}:
                upper_anchor = mode.upper_fn(
                    candidate_graph,
                    f_mhz,
                    pipeline_config_interval,
                    None,
                    mode.anchor_spec,
                )
                upper_spec = mode.upper_fn(
                    candidate_graph,
                    f_mhz,
                    pipeline_config_interval,
                    None,
                    mode.spec_spec,
                )
                crossed = mode.planner.consider(
                    itr=break_count,
                    graph=candidate_graph,
                    f_mhz=f_mhz,
                    II=pipeline_config_interval,
                    group=None,
                    upper_anchor_fn=mode.upper_fn,
                    anchor_spec=mode.anchor_spec,
                    upper_spec_fn=mode.upper_fn,
                    spec_spec=mode.spec_spec,
                    mean_predict=cached_mean_predict,
                    iteration=iteration_count,
                )
                # The baseline has an infinite cap and therefore never crosses.
                stopped_here = (
                    stop_on_crossing
                    and math.isfinite(mode.planner.cfg.cap_mW)
                    and crossed
                )
                effective_upper = upper_spec
            else:
                upper_robust = robust_upper_bound(pred, mode.robust_spec)
                snap = CandidateSnap(
                    itr=int(break_count),
                    f_mhz=float(f_mhz),
                    power_mean_mW=float(pred.total_mW),
                    power_upper_mW=float(upper_robust),
                    feat=feat_vec(candidate_graph, f_mhz),
                    robust_slack_mW=power_cap_mW - upper_robust,
                    iteration=int(iteration_count),
                )
                mode.planner.consider(snap)
                stopped_here = (
                    stop_on_crossing and upper_robust > power_cap_mW
                )
                effective_upper = upper_robust

            if stopped_here:
                mode.active = False
                mode.stopped_at_breaks = int(break_count)
                mode.stopped_at_iteration = int(iteration_count)
                mode.crossing_upper_mW = float(effective_upper)
                print(
                    f"  [all-modes] {mode.label} stopped at "
                    f"iteration={iteration_count}, breaks={break_count}: "
                    f"U={effective_upper:.3f} mW "
                    f"> cap={power_cap_mW:.3f} mW"
                )
            controller_elapsed_s[mode.key] = (
                time.perf_counter() - controller_start
            )

            log_trace_row(
                mode,
                iteration_count,
                break_count,
                f_mhz,
                pred,
                upper_anchor,
                upper_spec,
                upper_robust,
                stopped_here,
            )

        return {
            "predictor_s": predictor_elapsed_s,
            "controller_s": controller_elapsed_s,
            "evaluation_total_s": time.perf_counter() - evaluation_start,
        }

    def record_candidate_timing(timing_result, active_mode_keys, common_s):
        nonlocal predictor_total_s, predictor_calls
        predictor_total_s += float(timing_result["predictor_s"])
        predictor_calls += 1
        for mode_key, elapsed_s in timing_result["controller_s"].items():
            controller_total_s[mode_key] += float(elapsed_s)
        for mode_key in active_mode_keys:
            if mode_key not in timing_result["controller_s"]:
                continue
            mode_compile_time_s[mode_key] += (
                float(common_s)
                + float(timing_result["predictor_s"])
                + float(timing_result["controller_s"][mode_key])
            )

    # Evaluate the zero-break (no breaks in the critical path) candidate for every mode.
    search_loop_start = time.perf_counter()
    iteration_count = 0
    break_count = 0
    initial_active_mode_keys = [mode.key for mode in modes if mode.active]
    initial_timing = evaluate_candidate(
        iteration_count,
        break_count,
        graph,
        curr_freq,
    )
    record_candidate_timing(
        initial_timing,
        initial_active_mode_keys,
        common_s=0.0,
    )

    requested = os.environ.get("POST_PNR_ITR", "max")
    if requested == "max":
        max_breaks = None
    else:
        max_breaks = int(requested)
    explicit_limit = os.environ.get("CAPSTONE_MAX_BREAKS", "").strip()
    if explicit_limit:
        explicit_limit = int(explicit_limit)
        max_breaks = (
            explicit_limit
            if max_breaks is None
            else min(max_breaks, explicit_limit)
        )

    trajectory_end_reason = "all modes stopped"
    while any(mode.active for mode in modes):
        if max_breaks is not None and break_count >= max_breaks:
            trajectory_end_reason = f"configured limit of {max_breaks} breaks"
            break
        try:
            iteration_wall_start = time.perf_counter()
            pipelining_start = time.perf_counter()
            break_crit_path(
                graph,
                id_to_name,
                crit_path,
                placement,
                routing,
            )
            graph.regs = None
            update_kernel_latencies(
                app_dir,
                graph,
                id_to_name,
                placement,
                routing,
                existing_kernel_latencies,
                harden_flush,
                instance_to_instr,
                pipeline_config_interval,
                pes_with_packed_ponds,
                sparse,
            )
            pipelining_elapsed_s = time.perf_counter() - pipelining_start
            iteration_count += 1
            break_count += 1
            print(
                f"\nIteration {iteration_count} "
                f"(critical-path breaks={break_count}) frequency"
            )
            sta_start = time.perf_counter()
            curr_freq, crit_path, _ = sta(graph)
            sta_elapsed_s = time.perf_counter() - sta_start
            active_mode_keys = [mode.key for mode in modes if mode.active]
            candidate_timing = evaluate_candidate(
                iteration_count,
                break_count,
                graph,
                curr_freq,
            )
            iteration_elapsed_s = time.perf_counter() - iteration_wall_start

            pipelining_total_s += pipelining_elapsed_s
            sta_total_s += sta_elapsed_s
            post_pnr_iterations_total_s += iteration_elapsed_s
            successful_iterations += 1
            record_candidate_timing(
                candidate_timing,
                active_mode_keys,
                common_s=pipelining_elapsed_s + sta_elapsed_s,
            )
        except Exception as error:
            error_text = str(error)
            expected_exhaustion = (
                isinstance(error, ValueError)
                and (
                    "Can't find available register" in error_text
                    or "Cannot find available register" in error_text
                    or "critical path" in error_text.lower()
                )
            )
            if expected_exhaustion:
                trajectory_end_reason = "maximal pipeline candidate reached"
                print(f"\n[all-modes] {trajectory_end_reason}")
                if (
                    os.environ.get(
                        "CAPSTONE_TRACEBACK_ON_EXHAUSTION",
                        "0",
                    )
                    == "1"
                ):
                    traceback.print_exc()
            else:
                trajectory_end_reason = (
                    f"pipeline candidate generation failed: {error_text}"
                )
                print(f"\n[all-modes] {trajectory_end_reason}")
                traceback.print_exc()
            break

    search_loop_total_s = time.perf_counter() - search_loop_start

    selected_by_key = {}
    chosen_by_key = {}
    for mode in modes:
        chosen = mode.planner.finalize()
        chosen_by_key[mode.key] = chosen
        selected_by_key[mode.key] = chosen[0] if chosen else None

    baseline_candidate = selected_by_key.get("baseline")
    full_candidate = selected_by_key.get("capstone_iii_full")
    baseline_freq = (
        baseline_candidate.f_mhz if baseline_candidate is not None else None
    )
    full_freq = full_candidate.f_mhz if full_candidate is not None else None

    print("\n" + "=" * 78)
    print("SIMULTANEOUS CAPSTONE RESULTS")
    print(f"Kernel: {kernel}")
    print(f"Trajectory end: {trajectory_end_reason}")
    print(
        f"Trajectory totals: iterations={iteration_count}, "
        f"critical-path breaks={break_count}, "
        f"candidates evaluated={predictor_calls}"
    )
    print("=" * 78)
    for mode in modes:
        selected = selected_by_key[mode.key]
        if selected is None:
            detail = ""
            if isinstance(mode.planner, OnlinePlanner) and mode.planner.specs:
                detail = (
                    "; speculative candidates were ignored because no "
                    "safe anchor was established"
                )
            print(
                f"{mode.label}: NO SAFE CANDIDATE "
                f"(evaluated {mode.evaluated_candidates}{detail})"
            )
            continue
        status = (
            f"stopped at iteration={mode.stopped_at_iteration}, "
            f"breaks={mode.stopped_at_breaks}"
            if mode.stopped_at_breaks is not None
            else trajectory_end_reason
        )
        selected_iteration = (
            selected.itr
            if selected.iteration is None
            else selected.iteration
        )
        print(
            f"{mode.label}: selected iteration={selected_iteration}, "
            f"breaks={selected.itr}, "
            f"f={selected.f_mhz:.3f} MHz, "
            f"mean={selected.power_mean_mW:.3f} mW, "
            f"U={selected.power_upper_mW:.3f} mW; {status}"
        )
        for index, candidate in enumerate(chosen_by_key[mode.key]):
            candidate_iteration = (
                candidate.itr
                if candidate.iteration is None
                else candidate.iteration
            )
            print(
                f"  [{index}] {candidate.tag or 'CANDIDATE'} "
                f"iteration={candidate_iteration}, "
                f"breaks={candidate.itr}, f={candidate.f_mhz:.3f} MHz, "
                f"mean={candidate.power_mean_mW:.3f} mW, "
                f"U={candidate.power_upper_mW:.3f} mW"
            )

    table8_unpruned_k = int(
        os.environ.get(
            "CAPSTONE_TABLE8_UNPRUNED_K",
            os.environ.get("CAPSTONE_TABLE7_UNPRUNED_K", "90"),
        )
    )
    table8_pruned_k = _parse_positive_int_list(
        os.environ.get(
            "CAPSTONE_TABLE8_PRUNED_K",
            os.environ.get("CAPSTONE_TABLE7_PRUNED_K", "8,4"),
        ),
        default=(8, 4),
    )
    table8_rows = []
    for bound_mode in TABLE8_BOUND_MODE_ORDER:
        mode = mode_by_key[f"capstone_iii_{bound_mode}"]
        k_values = [table8_unpruned_k]
        if bound_mode == "full":
            k_values.extend(
                value
                for value in table8_pruned_k
                if value != table8_unpruned_k
            )
        for k_value in k_values:
            retained = select_top_bounded_candidates(
                mode.planner.safe_candidates,
                k_value,
            )
            selected = retained[0] if retained else None
            table8_rows.append(
                {
                    "bound_mode": bound_mode,
                    "K": int(k_value),
                    "retained_count": len(retained),
                    "selected": selected,
                }
            )

    print("\nTABLE VIII CANDIDATE-SELECTION RESULTS")
    for row in table8_rows:
        selected = row["selected"]
        if selected is None:
            result = "NO SAFE CANDIDATE"
        else:
            selected_iteration = (
                selected.itr
                if selected.iteration is None
                else selected.iteration
            )
            result = (
                f"iteration={selected_iteration}, breaks={selected.itr}, "
                f"f={selected.f_mhz:.3f} MHz, "
                f"mean={selected.power_mean_mW:.3f} mW, "
                f"U={selected.power_upper_mW:.3f} mW"
            )
        print(
            f"  {row['bound_mode']:<20} K={row['K']:<3} "
            f"retained={row['retained_count']:<3} {result}"
        )

    def average(total_s, count):
        return float(total_s) / float(count) if count else 0.0

    predictor_post_pnr_s = max(
        0.0,
        predictor_total_s - float(initial_timing["predictor_s"]),
    )
    predictor_post_pnr_calls = max(0, predictor_calls - 1)
    signoff_time_raw = os.environ.get(
        "CAPSTONE_SIGNOFF_POWER_TIME_S",
        "",
    ).strip()
    signoff_power_s = (
        None if not signoff_time_raw else float(signoff_time_raw)
    )
    if signoff_power_s is not None and signoff_power_s < 0.0:
        raise ValueError("CAPSTONE_SIGNOFF_POWER_TIME_S cannot be negative.")

    figure11_mode_keys = (
        ("Baseline", "baseline"),
        ("I", "capstone_i"),
        ("II", "capstone_ii"),
        ("III", "capstone_iii_full"),
    )
    baseline_compile_s = mode_compile_time_s["baseline"]
    normalized_compile_time = {}
    for short_label, mode_key in figure11_mode_keys:
        normalized_compile_time[mode_key] = (
            0.0
            if baseline_compile_s <= 0.0
            else float(mode_compile_time_s[mode_key] / baseline_compile_s)
        )

    print("\n" + "=" * 78)
    print("FIGURE 11 TIMING DATA")
    print(
        "Measured on the shared all-modes search; per-mode compile times below "
        "are estimates accounting for component-level variations."
    )
    print(
        f"Successful post-PnR iterations: {successful_iterations}; "
        f"candidate evaluations (including iteration 0): {predictor_calls}"
    )
    print(
        f"  STA (timing): total={sta_total_s:.6f} s, "
        f"time/iter={average(sta_total_s, successful_iterations):.6f} s"
    )
    print(
        f"  Pipelining: total={pipelining_total_s:.6f} s, "
        f"time/iter={average(pipelining_total_s, successful_iterations):.6f} s"
    )
    print(
        f"  Capstone predictor: total={predictor_post_pnr_s:.6f} s, "
        f"time/iter={average(predictor_post_pnr_s, predictor_post_pnr_calls):.6f} s"
    )
    print(
        f"  Post-PnR iterations: total={post_pnr_iterations_total_s:.6f} s, "
        f"time/iter={average(post_pnr_iterations_total_s, successful_iterations):.6f} s"
    )
    print(f"  Pipeline search loop: total={search_loop_total_s:.6f} s")
    if signoff_power_s is None:
        print(
            "  Signoff power: not run by this script; set "
            "CAPSTONE_SIGNOFF_POWER_TIME_S to record an externally measured time."
        )
    else:
        print(f"  Signoff power: total={signoff_power_s:.6f} s (external)")

    print("  Normalized compile-time estimates (Baseline = 1.0):")
    for short_label, mode_key in figure11_mode_keys:
        print(
            f"    {short_label:<8} "
            f"time={mode_compile_time_s[mode_key]:.6f} s, "
            f"normalized={normalized_compile_time[mode_key]:.6f}"
        )
    print("=" * 78)

    def summary_row(mode_key, bound_mode, k_value, retained_count, selected):
        mode = mode_by_key[mode_key]
        norm_vs_baseline = (
            ""
            if selected is None
            or baseline_freq is None
            or baseline_freq == 0
            else float(selected.f_mhz / baseline_freq)
        )
        norm_vs_full = (
            ""
            if selected is None or full_freq is None or full_freq == 0
            else float(selected.f_mhz / full_freq)
        )
        return {
            "run_id": run_id,
            "kernel": kernel,
            "mode": mode_key,
            "bound_mode": bound_mode,
            "K": int(k_value),
            "retained_count": int(retained_count),
            "evaluated_candidates": int(mode.evaluated_candidates),
            "trajectory_iterations": int(iteration_count),
            "trajectory_breaks": int(break_count),
            "stopped_at_iteration": (
                ""
                if mode.stopped_at_iteration is None
                else int(mode.stopped_at_iteration)
            ),
            "stopped_at_breaks": (
                ""
                if mode.stopped_at_breaks is None
                else int(mode.stopped_at_breaks)
            ),
            "selected_iteration": (
                ""
                if selected is None
                else int(
                    selected.itr
                    if selected.iteration is None
                    else selected.iteration
                )
            ),
            "selected_breaks": (
                "" if selected is None else int(selected.itr)
            ),
            "selected_freq_mhz": (
                "" if selected is None else float(selected.f_mhz)
            ),
            "baseline_freq_mhz": (
                "" if baseline_freq is None else float(baseline_freq)
            ),
            "norm_freq_vs_baseline": norm_vs_baseline,
            "full_bounds_freq_mhz": (
                "" if full_freq is None else float(full_freq)
            ),
            "norm_freq_vs_full_bounds": norm_vs_full,
            "P_mean_mW": (
                "" if selected is None else float(selected.power_mean_mW)
            ),
            "P_upper_mW": (
                "" if selected is None else float(selected.power_upper_mW)
            ),
            "power_cap_mW": (
                ""
                if not math.isfinite(mode.planner.cfg.cap_mW)
                else float(mode.planner.cfg.cap_mW)
            ),
            "predicted_headroom_mW": (
                ""
                if selected is None
                or not math.isfinite(mode.planner.cfg.cap_mW)
                else float(power_cap_mW - selected.power_upper_mW)
            ),
            "calibration_source": (
                ""
                if mode_key != "capstone_ii"
                else conformal_metadata["calibration_json"] or ""
            ),
            "anchor_has_finite_conformal_guarantee": (
                ""
                if mode_key != "capstone_ii"
                else conformal_metadata[
                    "anchor_has_finite_conformal_guarantee"
                ]
            ),
            # These three fields require signoff aggregates (protected by NDA).
            "P_oracle_mW": "",
            "success": "",
            "delta_cap_pct": "",
        }

    if summary_csv:
        # One row per controller result.
        for mode in modes:
            chosen = chosen_by_key[mode.key]
            _append_csv_row(
                summary_csv,
                summary_row(
                    mode.key,
                    mode.bound_mode,
                    k_outputs,
                    len(chosen),
                    selected_by_key[mode.key],
                ),
            )
        # Additional top-K rows used directly by the Table VIII sweep.
        for row in table8_rows:
            _append_csv_row(
                summary_csv,
                summary_row(
                    f"capstone_iii_{row['bound_mode']}",
                    row["bound_mode"],
                    row["K"],
                    row["retained_count"],
                    row["selected"],
                ),
            )

    if bitstreams_csv:
        for mode in modes:
            for rank, candidate in enumerate(chosen_by_key[mode.key]):
                candidate_iteration = (
                    candidate.itr
                    if candidate.iteration is None
                    else candidate.iteration
                )
                _append_csv_row(
                    bitstreams_csv,
                    {
                        "run_id": run_id,
                        "kernel": kernel,
                        "mode": mode.key,
                        "bound_mode": mode.bound_mode,
                        "rank": int(rank),
                        "tag": candidate.tag or "CANDIDATE",
                        "iteration": int(candidate_iteration),
                        "breaks": int(candidate.itr),
                        "f_mhz": float(candidate.f_mhz),
                        "P_mean_mW": float(candidate.power_mean_mW),
                        "P_upper_mW": float(candidate.power_upper_mW),
                        "power_cap_mW": (
                            ""
                            if not math.isfinite(mode.planner.cfg.cap_mW)
                            else float(mode.planner.cfg.cap_mW)
                        ),
                        "predicted_headroom_mW": (
                            ""
                            if candidate.robust_slack_mW is None
                            or not math.isfinite(candidate.robust_slack_mW)
                            else float(candidate.robust_slack_mW)
                        ),
                    },
                )

    timing_row = {
        "run_id": run_id,
        "kernel": kernel,
        "timing_method": "shared_trajectory_component_accounting",
        "successful_post_pnr_iterations": int(successful_iterations),
        "candidate_evaluations_including_iteration_0": int(predictor_calls),
        "trajectory_iterations": int(iteration_count),
        "trajectory_breaks": int(break_count),
        "sta_total_s": float(sta_total_s),
        "sta_per_iteration_s": average(
            sta_total_s,
            successful_iterations,
        ),
        "sta_share_pct": (
            ""
            if post_pnr_iterations_total_s <= 0.0
            else 100.0 * sta_total_s / post_pnr_iterations_total_s
        ),
        "pipelining_total_s": float(pipelining_total_s),
        "pipelining_per_iteration_s": average(
            pipelining_total_s,
            successful_iterations,
        ),
        "pipelining_share_pct": (
            ""
            if post_pnr_iterations_total_s <= 0.0
            else 100.0 * pipelining_total_s / post_pnr_iterations_total_s
        ),
        "capstone_predictor_total_s": float(predictor_post_pnr_s),
        "capstone_predictor_per_iteration_s": average(
            predictor_post_pnr_s,
            predictor_post_pnr_calls,
        ),
        "capstone_predictor_share_pct": (
            ""
            if post_pnr_iterations_total_s <= 0.0
            else 100.0 * predictor_post_pnr_s
            / post_pnr_iterations_total_s
        ),
        "post_pnr_iterations_total_s": float(
            post_pnr_iterations_total_s
        ),
        "post_pnr_iteration_mean_s": average(
            post_pnr_iterations_total_s,
            successful_iterations,
        ),
        "pipeline_search_loop_total_s": float(search_loop_total_s),
        "signoff_power_s": (
            "" if signoff_power_s is None else float(signoff_power_s)
        ),
        "baseline_compile_estimate_s": float(
            mode_compile_time_s["baseline"]
        ),
        "capstone_i_compile_estimate_s": float(
            mode_compile_time_s["capstone_i"]
        ),
        "capstone_ii_compile_estimate_s": float(
            mode_compile_time_s["capstone_ii"]
        ),
        "capstone_iii_compile_estimate_s": float(
            mode_compile_time_s["capstone_iii_full"]
        ),
        "baseline_normalized_compile_time": float(
            normalized_compile_time["baseline"]
        ),
        "capstone_i_normalized_compile_time": float(
            normalized_compile_time["capstone_i"]
        ),
        "capstone_ii_normalized_compile_time": float(
            normalized_compile_time["capstone_ii"]
        ),
        "capstone_iii_normalized_compile_time": float(
            normalized_compile_time["capstone_iii_full"]
        ),
        "controller_total_s_json": json.dumps(
            controller_total_s,
            sort_keys=True,
        ),
    }
    if timing_csv:
        _append_csv_row(timing_csv, timing_row)

    manifest = {
        "run_id": run_id,
        "kernel": kernel,
        "power_cap_mW": power_cap_mW,
        "trajectory_end_reason": trajectory_end_reason,
        "trajectory": {
            "iterations": int(iteration_count),
            "critical_path_breaks": int(break_count),
            "candidates_evaluated": int(predictor_calls),
            "successful_post_pnr_iterations": int(
                successful_iterations
            ),
        },
        "guardband": {
            "anchor_gamma": gb_anchor.gamma,
            "spec_gamma": gb_spec.gamma,
        },
        "power_model_scaling": {
            "enabled": power_scales["use_freq_ii_scaling"],
            "freq_ref_mhz": power_scales["freq_ref_mhz"],
            "II_ref": power_scales["II_ref"],
            "raw_pipeline_config_interval": power_scales[
                "raw_pipeline_config_interval"
            ],
        },
        "conformal": conformal_metadata,
        "selected_modes": {
            mode.key: _candidate_to_dict(selected_by_key[mode.key])
            for mode in modes
        },
        "final_bitstreams": {
            mode.key: [
                _candidate_to_dict(candidate)
                for candidate in chosen_by_key[mode.key]
            ]
            for mode in modes
        },
        "figure11_timing": timing_row,
        "table8": [
            {
                "bound_mode": row["bound_mode"],
                "K": row["K"],
                "retained_count": row["retained_count"],
                "selected": _candidate_to_dict(row["selected"]),
            }
            for row in table8_rows
        ],
    }
    if selection_json:
        os.makedirs(os.path.dirname(selection_json) or ".", exist_ok=True)
        with open(selection_json, "w") as output:
            json.dump(manifest, output, indent=2)

    print("\n[all-modes] Result files:")
    for description, path in (
        ("candidate trace CSV", trace_csv),
        ("controller summary CSV", summary_csv),
        ("final-bitstream CSV", bitstreams_csv),
        ("Figure 11 timing CSV", timing_csv),
        ("selection manifest JSON", selection_json),
    ):
        if path:
            print(f"  {description}: {os.path.abspath(path)}")

    primary_key = os.environ.get(
        "CAPSTONE_PRIMARY_OUTPUT_MODE",
        "capstone_iii_full",
    ).strip()
    if primary_key not in selected_by_key:
        raise ValueError(
            f"Unknown CAPSTONE_PRIMARY_OUTPUT_MODE={primary_key}. "
            f"Choose one of: {', '.join(selected_by_key)}"
        )
    primary = selected_by_key[primary_key]
    if primary is None:
        fallback_keys = ("capstone_iii_full", "capstone_ii", "capstone_i", "baseline")
        primary_key = next(
            (
                key
                for key in fallback_keys
                if selected_by_key.get(key) is not None
            ),
            None,
        )
        if primary_key is None:
            raise RuntimeError("No mode produced a candidate to emit.")
        primary = selected_by_key[primary_key]
        print(
            f"[all-modes] Requested primary mode had no safe candidate; "
            f"falling back to {primary_key}."
        )

    print(
        f"\n[all-modes] Standard design.route/design.place output uses "
        f"{primary_key}: iteration="
        f"{primary.itr if primary.iteration is None else primary.iteration}, "
        f"breaks={primary.itr}, f={primary.f_mhz:.3f} MHz, "
        f"mean={primary.power_mean_mW:.3f} mW, "
        f"U={primary.power_upper_mW:.3f} mW"
    )
    return _replay_post_pnr_candidate(
        app_dir=app_dir,
        break_count=primary.itr,
        placement_save=placement_save,
        routing_save=routing_save,
        id_to_name_save=id_to_name_save,
        netlist=netlist,
        pe_cycles=pe_cycles,
        io_cycles=io_cycles,
        existing_kernel_latencies=replay_reference_latencies,
        harden_flush=harden_flush,
        instance_to_instr=instance_to_instr,
        pipeline_config_interval=pipeline_config_interval,
        pes_with_packed_ponds=pes_with_packed_ponds,
        sparse=sparse,
    )


def pipeline_pnr(
    app_dir,
    placement,
    routing,
    id_to_name,
    netlist,
    load_only,
    harden_flush,
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    t_start = time.perf_counter()
    if load_only:
        packed_file = os.path.join(app_dir, "design.packed")
        id_to_name = pythunder.io.load_id_to_name(packed_file)
        return placement, routing, id_to_name

    placement_save = copy.deepcopy(placement)
    routing_save = copy.deepcopy(routing)
    id_to_name_save = copy.deepcopy(id_to_name)

    existing_kernel_latencies = {}
    if not sparse:
        kernel_latencies_file = glob.glob(f"{app_dir}/*_compute_kernel_latencies.json")[0]
        existing_kernel_latencies = json.load(open(kernel_latencies_file, "r"))

    if "PIPELINED" in os.environ and os.environ["PIPELINED"].isnumeric():
        pe_cycles = int(os.environ["PIPELINED"])
    else:
        pe_cycles = 1

    if "IO_DELAY" in os.environ and os.environ["IO_DELAY"] == "0":
        io_cycles = 0
    else:
        io_cycles = 1

    graph = construct_graph(
        placement,
        routing,
        id_to_name,
        netlist,
        pe_latency=pe_cycles,
        pond_latency=0,
        io_latency=io_cycles,
        sparse=sparse,
    )

    print("\nApplication Frequency:")
    curr_freq, crit_path, crit_nets = sta(graph)

    update_kernel_latencies(
        app_dir,
        graph,
        id_to_name,
        placement,
        routing,
        existing_kernel_latencies,
        harden_flush,
        instance_to_instr,
        pipeline_config_interval,
        pes_with_packed_ponds,
        sparse,
    )

    run_all_modes = os.environ.get("CAPSTONE_RUN_ALL_MODES", "1") != "0"
    if "POST_PNR_ITR" in os.environ and run_all_modes:
        (
            graph,
            curr_freq,
            crit_path,
            crit_nets,
            placement,
            routing,
            id_to_name,
        ) = run_all_capstone_modes_post_pnr(
            app_dir=app_dir,
            graph=graph,
            curr_freq=curr_freq,
            crit_path=crit_path,
            placement=placement,
            routing=routing,
            id_to_name=id_to_name,
            placement_save=placement_save,
            routing_save=routing_save,
            id_to_name_save=id_to_name_save,
            netlist=netlist,
            pe_cycles=pe_cycles,
            io_cycles=io_cycles,
            existing_kernel_latencies=existing_kernel_latencies,
            harden_flush=harden_flush,
            instance_to_instr=instance_to_instr,
            pipeline_config_interval=pipeline_config_interval,
            pes_with_packed_ponds=pes_with_packed_ponds,
            sparse=sparse,
        )
    elif "POST_PNR_ITR" in os.environ:
        if os.environ["POST_PNR_ITR"] == "max":
            max_itr = None
        else:
            max_itr = int(os.environ["POST_PNR_ITR"])

        # Keep the baseline STA result for Capstone III's break_count=0 candidate.
        initial_freq = curr_freq
        curr_freq = initial_freq
        itr = 0
        
        # 0) Units: cap in mW. Configure with CAPSTONE_POWER_CAP_MW.
        power_cap_mW = float(os.environ.get("CAPSTONE_POWER_CAP_MW", "96.0"))
        K_outputs = int(os.environ.get("NUM_BITSTREAMS", "4"))
        model_dir = os.environ.get("CAPSTONE_MODEL_DIR", ".")
        power_scales = make_power_model_scales(
            model_dir,
            pipeline_config_interval,
        )

        # 1) Wrap mean model (predict_power) to have a (graph,f,II) -> mW signature.
        def mean_predict(graph, f_mhz, II):
            return predict_power(
                graph,
                f_mhz,
                instance_to_instr,
                II,
                scales=power_scales,
            )

        planner_mode = os.environ.get("CAPSTONE_MODE", "bounded").lower()
        assert planner_mode in {"guardband", "conformal", "bounded"}, (
            f"Unknown CAPSTONE_MODE={planner_mode}"
        )

        # 2A) Conformal mode
        if planner_mode == "conformal":
            (
                conf_anchor,
                conf_spec,
                conformal_metadata,
            ) = configure_capstone_ii_conformal_specs()
            print(
                "[capstone] Capstone II global calibration: "
                f"n={conformal_metadata['n_global']}, "
                f"spec={conformal_metadata['spec_source']}, "
                f"anchor={conformal_metadata['anchor_source']}"
            )
        else:
            # These placeholders are never selected outside conformal mode.
            conf_anchor = ConformalSpec(
                residual_q_mW={"global": float("inf")},
                alpha=0.005,
            )
            conf_spec = ConformalSpec(
                residual_q_mW={"global": float("inf")},
                alpha=0.05,
            )
        U_conf = make_conformal_upper(mean_predict) # upper bound

        # 2B) Guardband mode (simple)
        gb_anchor = GuardbandSpec(
            gamma=float(os.environ.get("CAPSTONE_I_ANCHOR_GAMMA", "0.45"))
        )
        gb_spec = GuardbandSpec(
            gamma=float(os.environ.get("CAPSTONE_I_SPEC_GAMMA", "0.30"))
        )
        U_gb      = make_guardband_upper(mean_predict) # upper bound

        # Choose which we want to use:
        use_conformal = (planner_mode == "conformal")
        upper_anchor_fn = U_conf if use_conformal else U_gb
        upper_spec_fn   = U_conf if use_conformal else U_gb
        anchor_spec = conf_anchor if use_conformal else gb_anchor
        spec_spec   = conf_spec   if use_conformal else gb_spec

        # If we have workload/application kernel groups, pass a 'group' string below; else use None
        def group_for_graph(graph):
            return None  # or e.g., graph.kernel_name
        
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        if planner_mode != "bounded":
            planner = OnlinePlanner(OnlinePlanConfig(
                cap_mW=power_cap_mW,
                K_outputs=K_outputs,
                stop_on="spec", # stop when the speculative upper bound crosses the power cap
                diversity_weight=0.01,
                min_delta_freq_MHz=1.0
            ))
            
            while max_itr == None:
                try:
                    break_crit_path(graph, id_to_name, crit_path, placement, routing)
                    graph.regs = None
                    update_kernel_latencies(
                        app_dir,
                        graph,
                        id_to_name,
                        placement,
                        routing,
                        existing_kernel_latencies,
                        harden_flush,
                        instance_to_instr,
                        pipeline_config_interval,
                        pes_with_packed_ponds,
                        sparse,
                    )

                    print("\nIteration", itr + 1, "frequency")
                    curr_freq, crit_path, crit_nets = sta(graph)
                    curr_power_mean = mean_predict(graph, curr_freq, pipeline_config_interval) # raw power prediction before applying any upper bound
                    
                    # call planner (note: conformal function needs 'group'; guardband ignores it)
                    stop_now = planner.consider(
                        itr=itr + 1, graph=graph, f_mhz=curr_freq, II=pipeline_config_interval,
                        group=group_for_graph(graph),
                        upper_anchor_fn=upper_anchor_fn, anchor_spec=anchor_spec,
                        upper_spec_fn=upper_spec_fn,     spec_spec=spec_spec,
                        mean_predict=mean_predict
                    )
                    print(f"  Power mean = {curr_power_mean:.3f} mW")
                    print("planner_mode: ", planner_mode)
                    if planner.anchor: # U_anchor is the upper bound power value of the anchor configuration
                        print(f"  Anchor @ breaks={planner.anchor.itr}: U_anchor={planner.anchor.power_upper_mW:.3f} mW, f={planner.anchor.f_mhz:.1f} MHz")
                    if planner.specs:  # U_spec is the upper bound power value of the speculative configuration
                        best = planner.specs[0]
                        print(f"  Best spec  @ breaks={best.itr}: U_spec  ={best.power_upper_mW:.3f} mW, f={best.f_mhz:.1f} MHz")

                    if stop_now:
                        print("* * * Upper bound over cap. Stopping.")
                        chosen = planner.finalize()
                        # Guarantee at least the anchor:
                        if not chosen:
                            raise RuntimeError("Planner produced no candidates; tune envelopes.")
                        # Replay the top choice (anchor-first)
                        max_itr = chosen[0].itr
                        selected = chosen
                        break

                except Exception as e:
                    print("Pipeline iteration failed:", e)
                    traceback.print_exc()
                    max_itr = itr
                itr += 1

            print("\nCan break", max_itr, "critical paths")
            chosen = planner.finalize()
            print("\nChosen bitstreams (anytime-safe set):")
            for i, c in enumerate(chosen):
                tag = "ANCHOR" if planner.anchor and c.itr == planner.anchor.itr else "SPEC"
                print(f"  [{i}] {tag}: breaks={c.itr}, f={c.f_mhz:.1f} MHz, U={c.power_upper_mW:.3f} mW, mean={c.power_mean_mW:.3f} mW")

            # Emit each chosen candidate
            for i, c in enumerate(chosen):
                # reload baseline
                id_to_name = id_to_name_save
                placement  = placement_save
                routing    = routing_save
                graph = construct_graph(placement, routing, id_to_name, netlist,
                                        pe_latency=pe_cycles, pond_latency=0, io_latency=io_cycles, sparse=sparse)
                # re-run latency update once
                update_kernel_latencies(app_dir, graph, id_to_name, placement, routing,
                                        existing_kernel_latencies, harden_flush,
                                        instance_to_instr, pipeline_config_interval,
                                        pes_with_packed_ponds, sparse)
                # apply c.itr breaks
                for _ in range(c.itr):
                    f, crit_path, _ = sta(graph)
                    break_crit_path(graph, id_to_name, crit_path, placement, routing)

                # (optional) re-run STA to record final f
                f_final, _, _ = sta(graph)

                # dump per-candidate outputs
                dump_routing_result(app_dir, routing)
                dump_placement_result(app_dir, placement, id_to_name)
                with open(os.path.join(app_dir, f"design.{i}.freq"), "w") as fh:
                    fh.write(f"{f_final}\n")

            # We can now recreate each chosen design by reloading/snapshooting the state at iteration 'c.itr'
            # (already have a reload block of code; can extend it to capture snapshots per itr.)
            
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        else:
            # Capstone III bounded-error mode:
            # 1. Build one robust upper envelope U_rob from event-level epsilons.
            # 2. Keep the highest-frequency candidate with U_rob <= cap as the anchor.
            # 3. Optionally return up to K-1 Pareto-pruned, diverse candidates.
            robust_spec = load_robust_bound_spec(model_dir=model_dir)
            bounded_planner = BoundedPlanner(OnlinePlanConfig(
                cap_mW=power_cap_mW,
                K_outputs=K_outputs,
                stop_on="anchor",
                diversity_weight=float(os.environ.get("CAPSTONE_DIVERSITY_WEIGHT", "0.01")),
                min_delta_freq_MHz=float(os.environ.get("CAPSTONE_MIN_DELTA_FREQ_MHZ", "1.0")),
            ))

            def consider_bounded_candidate(break_count, curr_graph, curr_freq):
                pred = predict_power_components(
                    curr_graph,
                    curr_freq,
                    instance_to_instr,
                    pipeline_config_interval,
                    scales=power_scales,
                )
                U_rob = robust_upper_bound(pred, robust_spec)

                snap = CandidateSnap(
                    itr=break_count,
                    f_mhz=float(curr_freq),
                    power_mean_mW=pred.total_mW,
                    power_upper_mW=U_rob,
                    feat=feat_vec(curr_graph, curr_freq),
                    robust_slack_mW=power_cap_mW - U_rob,
                )
                return bounded_planner.consider(snap)

            # Include the already-routed baseline candidate. This matters if the first
            # post-PnR pipelined candidate already exceeds the cap.
            consider_bounded_candidate(0, graph, curr_freq)

            while max_itr is None:
                try:
                    break_crit_path(graph, id_to_name, crit_path, placement, routing)
                    graph.regs = None
                    update_kernel_latencies(
                        app_dir,
                        graph,
                        id_to_name,
                        placement,
                        routing,
                        existing_kernel_latencies,
                        harden_flush,
                        instance_to_instr,
                        pipeline_config_interval,
                        pes_with_packed_ponds,
                        sparse,
                    )

                    break_count = itr + 1
                    print("\nIteration", break_count, "frequency")
                    curr_freq, crit_path, crit_nets = sta(graph)

                    stop_now = consider_bounded_candidate(break_count, graph, curr_freq)
                    if stop_now:
                        print("* * * Robust upper bound is over cap. Stopping bounded search.")
                        break

                except Exception as e:
                    print("Pipeline iteration failed:", e)
                    traceback.print_exc()
                    break
                itr += 1

            chosen = bounded_planner.finalize()
            if not chosen:
                raise RuntimeError(
                    "Capstone III produced no robust-safe candidate. "
                    "Increase CAPSTONE_POWER_CAP_MW, reduce CAPSTONE_III_DEFAULT_REL_EPS, "
                    "or provide less conservative CAPSTONE_BOUNDS_JSON."
                )
            max_itr = chosen[0].itr

            print("\nChosen bitstreams (Capstone III bounded-error set):")
            for i, c in enumerate(chosen):
                tag = "ANCHOR" if i == 0 else "SPEC"
                print(f"  [{i}] {tag}: breaks={c.itr}, f={c.f_mhz:.1f} MHz, "
                      f"U_rob={c.power_upper_mW:.3f} mW, mean={c.power_mean_mW:.3f} mW, "
                      f"headroom={c.robust_slack_mW:.3f} mW")

            # Emit each selected bounded candidate, not just the final max_itr design.
            for i, c in enumerate(chosen):
                id_to_name = copy.deepcopy(id_to_name_save)
                placement = copy.deepcopy(placement_save)
                routing = copy.deepcopy(routing_save)
                graph = construct_graph(
                    placement,
                    routing,
                    id_to_name,
                    netlist,
                    pe_latency=pe_cycles,
                    pond_latency=0,
                    io_latency=io_cycles,
                    sparse=sparse,
                )
                update_kernel_latencies(
                    app_dir,
                    graph,
                    id_to_name,
                    placement,
                    routing,
                    existing_kernel_latencies,
                    harden_flush,
                    instance_to_instr,
                    pipeline_config_interval,
                    pes_with_packed_ponds,
                    sparse,
                )
                for _ in range(c.itr):
                    f_tmp, crit_path_tmp, _ = sta(graph)
                    break_crit_path(graph, id_to_name, crit_path_tmp, placement, routing)

                update_kernel_latencies(
                    app_dir,
                    graph,
                    id_to_name,
                    placement,
                    routing,
                    existing_kernel_latencies,
                    harden_flush,
                    instance_to_instr,
                    pipeline_config_interval,
                    pes_with_packed_ponds,
                    sparse,
                )
                f_final, _, _ = sta(graph)
                dump_routing_result(app_dir, routing)
                dump_placement_result(app_dir, placement, id_to_name)
                with open(os.path.join(app_dir, f"design.{i}.freq"), "w") as fh:
                    fh.write(f"{f_final}\n")

        if planner_mode == "bounded":
            # Bounded-mode candidates were printed/emitted in the bounded branch above.
            pass
        else:
            print("\nChosen bitstreams (anytime-safe set):")
            for i, c in enumerate(chosen):
                tag = "ANCHOR" if planner.anchor and c.itr == planner.anchor.itr else "SPEC"
                print(f"  [{i}] {tag}: breaks={c.itr}, f={c.f_mhz:.1f} MHz, "
                    f"U={c.power_upper_mW:.3f} mW, mean={c.power_mean_mW:.3f} mW")

        # Reloading best result
        id_to_name = id_to_name_save
        placement = placement_save
        routing = routing_save
        graph = construct_graph(
            placement,
            routing,
            id_to_name,
            netlist,
            pe_latency=pe_cycles,
            pond_latency=0,
            io_latency=io_cycles,
            sparse=sparse,
        )
        starting_regs = graph.added_regs

        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            existing_kernel_latencies,
            harden_flush,
            instance_to_instr,
            pipeline_config_interval,
            pes_with_packed_ponds,
            sparse,
        )

        for _ in range(max_itr):
            curr_freq, crit_path, crit_nets = sta(graph)
            break_crit_path(graph, id_to_name, crit_path, placement, routing)

        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            existing_kernel_latencies,
            harden_flush,
            instance_to_instr,
            pipeline_config_interval,
            pes_with_packed_ponds,
            sparse,
        )
        print("\nFinal application frequency:")
        curr_freq, crit_path, crit_nets = sta(graph)

        if max_itr == 0:
            print(bcolors.WARNING + "\nCouldn't break any paths" + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + "\nBroke", max_itr, "critical paths" + bcolors.ENDC)

        print(
            "\nAdded", graph.added_regs - starting_regs, "registers to routing graph\n"
        )
    elif "EXHAUSTIVE_PIPE" in os.environ:
        starting_regs = graph.added_regs
        exhaustive_pipe(graph, id_to_name, placement, routing)
        curr_freq, crit_path, crit_nets = sta(graph)
        print(
            "\nAdded", graph.added_regs - starting_regs, "registers to routing graph\n"
        )

    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_routing_result(app_dir, routing)
    dump_placement_result(app_dir, placement, id_to_name)

    t_end = time.perf_counter()
    print(f"[Timing] pipeline_pnr took {t_end - t_start:.2f} seconds.")

    return placement, routing, id_to_name
