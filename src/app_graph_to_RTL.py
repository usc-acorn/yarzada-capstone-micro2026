import os
from archipelago.pnr_graph import (TileType, RouteType, TileNode, RouteNode, RoutingResultGraph)
from collections import defaultdict

def sanitize_name(node):
    if isinstance(node, TileNode):
        return f"{node.tile_type.name.lower()}_{node.tile_id}"
    else:
        return f"{node.route_type.name.lower()}_{node.x}_{node.y}_{node.track}_{node.side}_{node.io}_{node.bit_width}_{node.port}_{node.net_id}_{node.reg_name}_{node.rmux_name}_{node.reg}_{node.kernel}" # unique name

def emit_route_inst(route, data_width=16):
    module_name = {
        "SB": "SB_route",
        "RMUX": "RMUX_route",
        "PORT": "PORT_route",
        "REG": "REG_route"
    }[route.route_type.name]

    inst_name = f"{route.route_type.name.lower()}_{route.x}_{route.y}_{route.track}_{route.side}_{route.io}_{route.bit_width}_{route.port}_{route.net_id}_{route.reg_name}_{route.rmux_name}_{route.reg}_{route.kernel}" # unique name

    ports = [
        f".clk(clk)",
        f".rst_n(rst_n)"
    ]
    
    ports.append(f".in_data(wire_{inst_name}_in_data)")
    ports.append(f".in_valid(wire_{inst_name}_in_valid)")
    ports.append(f".in_ready(wire_{inst_name}_in_ready)")
    ports.append(f".out_data(wire_{inst_name}_out_data)")
    ports.append(f".out_valid(wire_{inst_name}_out_valid)")
    ports.append(f".out_ready(wire_{inst_name}_out_ready)")
    return f"  {module_name} {inst_name} (\n      " + ",\n      ".join(ports) + "\n  );\n"

def emit_tile_inst(graph, tile, data_width=16):
    module_name = {
        'PE': 'pe_tile',
        'MEM': 'mem_tile',
        'REG': 'pipeline_reg',
        'SWITCH': 'switch_tile',
        'IO1': 'io_tile',
        'IO16': 'io_tile'
    }[tile.tile_type.name]
    
    inst = f"{tile.tile_type.name.lower()}_{tile.tile_id}"  # unique name

    inst_name = f"{tile.tile_type.name.lower()}_{tile.tile_id}"  # unique name

    ports = [
        f".clk(clk)",
        f".rst_n(rst_n)"
    ]
    
    if tile.tile_type.name in ('IO1', 'IO16', 'REG', 'SWITCH'):
        sid = tile.tile_id
        ports.append(f".in_data(wire_{inst_name}_in_data)")
        ports.append(f".in_valid(wire_{inst_name}_in_valid)")
        ports.append(f".in_ready(wire_{inst_name}_in_ready)")
        ports.append(f".out_data(wire_{inst_name}_out_data)")
        ports.append(f".out_valid(wire_{inst_name}_out_valid)")
        ports.append(f".out_ready(wire_{inst_name}_out_ready)")
        
    elif tile.tile_type.name in ('PE','MEM'):
        port0 = None
        port1 = None
        port2 = None
        port3 = None
        
        for in_port in tile.input_port_latencies:
            if in_port[-1] == '0':
                port0 = in_port

            elif in_port[-1] == '1':
                port1 = in_port
            elif in_port[-1] == '2':
                port2 = in_port
            elif in_port[-1] == '3':
                port3 = in_port
        if tile.tile_type.name == "PE":  
            ports.append(f".cfg_opcode(wire_pe_cfg_opcode)")
            ports.append(f".cfg_valid(wire_pe_cfg_valid)")

        # Compute operand_mask bits based on which ports are connected
        # bit 0 = in0, bit 1 = in1, etc.
        mask = 0
        if port0 is not None:
            mask |= 1 << 0  # bit 0
        if port1 is not None:
            mask |= 1 << 1  # bit 1
        if port2 is not None:
            mask |= 1 << 2  # bit 2
        if port3 is not None:
            mask |= 1 << 3  # bit 3
        ports.append(f"      .operand_mask(4'b{mask:04b})")
        
        if port0 is None:
            ports.append(f".in0()")
        else:
            ports.append(f".in0(wire_{inst_name}_in0_data)")
        if port1 is None:
            ports.append(f".in1()")
        else:
            ports.append(f".in1(wire_{inst_name}_in1_data)")
        if port2 is None:
            ports.append(f".in2()")
        else:
            ports.append(f".in2(wire_{inst_name}_in2_data)")
        if port3 is None:
            ports.append(f".in3()")
        else:
            ports.append(f".in3(wire_{inst_name}_in3_data)")
            
            
        if port0 is None:
            ports.append(f".in0_valid()")
        else:
            ports.append(f".in0_valid(wire_{inst_name}_in0_valid)")
        if port1 is None:
            ports.append(f".in1_valid()")
        else:
            ports.append(f".in1_valid(wire_{inst_name}_in1_valid)")
        if port2 is None:
            ports.append(f".in2_valid()")
        else:
            ports.append(f".in2_valid(wire_{inst_name}_in2_valid)")
        if port3 is None:
            ports.append(f".in3_valid()")
        else:
            ports.append(f".in3_valid(wire_{inst_name}_in3_valid)")            
            
        if port0 is None:
            ports.append(f".in0_ready()")
        else:
            ports.append(f".in0_ready(wire_{inst_name}_in0_ready)")
        if port1 is None:
            ports.append(f".in1_ready()")
        else:
            ports.append(f".in1_ready(wire_{inst_name}_in1_ready)")
        if port2 is None:
            ports.append(f".in2_ready()")
        else:
            ports.append(f".in2_ready(wire_{inst_name}_in2_ready)")
        if port3 is None:
            ports.append(f".in3_ready()")
        else:
            ports.append(f".in3_ready(wire_{inst_name}_in3_ready)") 
        
        # single output
        ports.append(f".out(wire_{inst_name}_out_data)")
        ports.append(f".out_valid(wire_{inst_name}_out_valid)")
        ports.append(f".out_ready(wire_{inst_name}_out_ready)")
            
    else:
        raise ValueError(f"Unsupported tile type: {tile.tile_type.name}")

    return f"  {module_name} {inst_name} (\n      " + ",\n      ".join(ports) + "\n  );\n"

def emit_wire_defs(graph, data_width):
    wires = set()
    for tile in graph.get_tiles():
        base = f"{tile.tile_type.name.lower()}_{tile.tile_id}"
        
        wires.add(f"logic [7:0]  wire_pe_cfg_opcode;")
        wires.add(f"logic        wire_pe_cfg_valid;")
        if tile.tile_type.name in ('IO1', 'IO16', 'REG', 'SWITCH'):
            wires.add(f"logic [{data_width-1}:0] wire_{base}_in_data;")
            wires.add(f"logic        wire_{base}_in_valid;")
            wires.add(f"logic        wire_{base}_in_ready;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_out_data;")
            wires.add(f"logic        wire_{base}_out_valid;")
            wires.add(f"logic        wire_{base}_out_ready;")
        
        if tile.tile_type.name in ('PE', 'MEM'):
            if tile.tile_type.name == "PE":  
                wires.add(f"logic [7:0] wire_{base}_cfg_opcode;")
                wires.add(f"logic        wire_{base}_cfg_valid;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_in0_data;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_in1_data;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_in2_data;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_in3_data;")
            wires.add(f"logic        wire_{base}_in0_valid;")
            wires.add(f"logic        wire_{base}_in1_valid;")
            wires.add(f"logic        wire_{base}_in2_valid;")
            wires.add(f"logic        wire_{base}_in3_valid;")
            wires.add(f"logic        wire_{base}_in0_ready;")
            wires.add(f"logic        wire_{base}_in1_ready;")
            wires.add(f"logic        wire_{base}_in2_ready;")
            wires.add(f"logic        wire_{base}_in3_ready;")
            wires.add(f"logic [{data_width-1}:0] wire_{base}_out_data;")
            wires.add(f"logic        wire_{base}_out_valid;")
            wires.add(f"logic        wire_{base}_out_ready;")

    for route in graph.get_routes():
        base = f"{route.route_type.name.lower()}_{route.x}_{route.y}_{route.track}_{route.side}_{route.io}_{route.bit_width}_{route.port}_{route.net_id}_{route.reg_name}_{route.rmux_name}_{route.reg}_{route.kernel}"
        wires.add(f"logic [{data_width-1}:0] wire_{base}_in_data;")
        wires.add(f"logic        wire_{base}_in_valid;")
        wires.add(f"logic        wire_{base}_in_ready;")
        wires.add(f"logic [{data_width-1}:0] wire_{base}_out_data;")
        wires.add(f"logic        wire_{base}_out_valid;")
        wires.add(f"logic        wire_{base}_out_ready;")

    return "\n".join(sorted(wires))

def emit_assigns(graph):

    assigns = []

    assign_buckets = defaultdict(list)
    other_lines   = []

    input_ios = sorted(graph.get_input_ios(), key=lambda t: t.tile_id)
    output_ios = sorted(graph.get_output_ios(), key=lambda t: t.tile_id)

    other_lines.append(f"assign wire_pe_cfg_opcode = pe_cfg_opcode;")
    other_lines.append(f"assign wire_pe_cfg_valid = pe_cfg_valid;")
    for io in graph.get_input_ios():
        sid = io.tile_id
        inst = sanitize_name(io) # e.g. "io16_12"
        other_lines.append(f"assign wire_{inst}_in_data  = in_data_{sid};")
        other_lines.append(f"assign wire_{inst}_in_valid = in_valid_{sid};")
        other_lines.append(f"assign in_ready_{sid}          = wire_{inst}_in_ready;")
        
    for io in graph.get_output_ios():
        sid  = io.tile_id
        inst = sanitize_name(io)
        # drive the io_tile’s out_ready from module port
        other_lines.append(f"assign wire_{inst}_out_ready = out_ready_{sid};")

    for src, dst in graph.edges:

        if hasattr(src, 'tile_type'): # src node is a TileNode
            src_inst_name = f"{src.tile_type.name.lower()}_{src.tile_id}"  # name of the corresponding module instantiation
            src_wire_data = f"wire_{src_inst_name}_out_data"
            src_wire_valid = f"wire_{src_inst_name}_out_valid"
            src_wire_ready = f"wire_{src_inst_name}_out_ready"
                
        elif hasattr(src, 'route_type'): # src node is a RouteNode
            src_inst_name = f"{src.route_type.name.lower()}_{src.x}_{src.y}_{src.track}_{src.side}_{src.io}_{src.bit_width}_{src.port}_{src.net_id}_{src.reg_name}_{src.rmux_name}_{src.reg}_{src.kernel}" # name of the corresponding module instantiation
            src_wire_data = f"wire_{src_inst_name}_out_data"
            src_wire_valid = f"wire_{src_inst_name}_out_valid"
            src_wire_ready = f"wire_{src_inst_name}_out_ready"
            if src.route_type.name == "PORT": 
                port = src.port 
                                
        if hasattr(dst, 'tile_type'): # dst node is a TileNode
            dst_inst_name = f"{dst.tile_type.name.lower()}_{dst.tile_id}"  # name of the corresponding module instantiation
            if dst.tile_type.name == "PE":   
                dst_wire_data = f"wire_{dst_inst_name}_in{port[-1]}_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in{port[-1]}_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in{port[-1]}_ready"
            elif dst.tile_type.name == "MEM":
                dst_wire_data = f"wire_{dst_inst_name}_in{port[-1]}_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in{port[-1]}_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in{port[-1]}_ready"
            elif dst.tile_type.name == "REG":
                dst_wire_data = f"wire_{dst_inst_name}_in_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in_ready"
            else:
                dst_wire_data = f"wire_{dst_inst_name}_in_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in_ready"

        elif hasattr(dst, 'route_type'): # dst node is a RouteNode
            dst_inst_name = f"{dst.route_type.name.lower()}_{dst.x}_{dst.y}_{dst.track}_{dst.side}_{dst.io}_{dst.bit_width}_{dst.port}_{dst.net_id}_{dst.reg_name}_{dst.rmux_name}_{dst.reg}_{dst.kernel}" # name of the corresponding module instantiation
            if dst.route_type.name == "PORT": 
                dst_wire_data = f"wire_{dst_inst_name}_in_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in_ready"

            else: 
                dst_wire_data = f"wire_{dst_inst_name}_in_data"
                dst_wire_valid = f"wire_{dst_inst_name}_in_valid"
                dst_wire_ready = f"wire_{dst_inst_name}_in_ready"

        assign_buckets[src_wire_ready].append(dst_wire_ready)

        other_lines.append(f"assign {dst_wire_data}  = {src_wire_data};")
        other_lines.append(f"assign {dst_wire_valid} = {src_wire_valid};")

    out = []
    for lhs, rhss in assign_buckets.items():
        if len(rhss) == 1:
            out.append(f"assign {lhs} = {rhss[0]};")
        else:
            out.append(f"assign {lhs} = " + " && ".join(rhss) + ";")

    out.extend(other_lines)

    return "\n".join(out)

    
def app_graph_to_RTL(graph, output_file="application_graph.v", data_width=16):
    input_ios = sorted(graph.get_input_ios(), key=lambda t: t.tile_id)
    output_ios = sorted(graph.get_output_ios(), key=lambda t: t.tile_id)

    tiles1 = graph.get_tiles()

    routes1 = graph.get_routes()

    mems1 = graph.get_mems()

    regs1 = graph.get_regs()

    pes1 = graph.get_pes()

    roms1 = graph.get_roms()

    ponds1 = graph.get_ponds()

    shift_regs1 = graph.get_shift_regs()

    with open(output_file, "w") as f:
        f.write("// Auto-generated application_graph (val/rdy)\n")
        f.write("module application_graph (\n")
        f.write("  input  logic        clk,\n")
        f.write("  input  logic        rst_n,\n")
        f.write("// Config: 0=ADD 1=MUL 2=MAC 3=SUB 4=RELU\n")
        f.write("  input  logic  [7:0] pe_cfg_opcode,\n")
        f.write("  input  logic        pe_cfg_valid,\n")
          
        for io in sorted(graph.get_input_ios(), key=lambda t: t.tile_id):
            f.write(f"  input  logic        in_valid_{io.tile_id},\n")
            f.write(f"  output logic        in_ready_{io.tile_id},\n")
            f.write(f"  input  logic [{data_width-1}:0] in_data_{io.tile_id},\n")
        output_ios = sorted(graph.get_output_ios(), key=lambda t: t.tile_id)
        for i, io in enumerate(output_ios):
            comma = ',' if i < len(output_ios) - 1 else ''
            f.write(f"  output logic [{data_width-1}:0] out_data_{io.tile_id},\n")
            f.write(f"  output logic        out_valid_{io.tile_id},\n")
            f.write(f"  input  logic        out_ready_{io.tile_id}{comma}\n")
        f.write(");\n\n")
        
        # Wire declarations
        f.write("  // Wires\n")
        f.write(emit_wire_defs(graph, data_width))
        f.write("\n\n")

        # TileNode instantiations
        f.write("  // TileNode instances\n")
        for tile in graph.get_tiles():
            f.write(emit_tile_inst(graph, tile, data_width))
            f.write("\n")

        # RouteNode instantiations
        f.write("  // RouteNode instances\n")
        for route in graph.get_routes():
            f.write(emit_route_inst(route, data_width))
            f.write("\n")

        # Assignments
        f.write("  // Wire connections\n")
        f.write(emit_assigns(graph))

        # Output preparation logic
        f.write("\n  // ---------------------------------------------------\n")
        f.write("  // Output preparation logic\n")
        f.write("  // ---------------------------------------------------\n")
        
        for io in output_ios:
            sid = io.tile_id
            f.write(f"  logic [{data_width-1}:0] out_{sid}_reg;\n")
            f.write(f"  logic        out_{sid}_valid_reg;\n")
 
        f.write("\n")

        f.write("  always_ff @(posedge clk or negedge rst_n) begin\n")
        f.write("    if (!rst_n) begin\n")

        for io in output_ios:
            sid = io.tile_id
            f.write(f"      out_{sid}_reg   <= 'b0;\n")
            f.write(f"      out_{sid}_valid_reg <= 1'b0;\n")

        f.write("    end else begin\n")

        for io in output_ios:
            sid = io.tile_id
            inst = sanitize_name(io)
            f.write(f"      if (wire_{inst}_out_valid) begin\n")
            f.write(f"        out_{sid}_reg   <= wire_{inst}_out_data;\n")
            f.write(f"        out_{sid}_valid_reg <= 1'b1;\n")
            f.write(f"      end\n")

        clear_conds = ' && '.join([f"out_ready_{io.tile_id} && out_{io.tile_id}_valid_reg" for io in output_ios])
        f.write(f"      if ({clear_conds}) begin\n")
        for io in output_ios:
            sid = io.tile_id
            f.write(f"        out_{sid}_valid_reg <= 1'b0;\n")
        f.write("      end\n")
        f.write("    end\n")
        f.write("  end\n\n")

        for io in output_ios:
            sid = io.tile_id
            f.write(f"  assign out_data_{sid}  = out_{sid}_reg;\n")
            f.write(f"  assign out_valid_{sid} = out_{sid}_valid_reg;\n")

        f.write("\nendmodule\n")

    print(f"Completed generating application graph RTL: {output_file}")
