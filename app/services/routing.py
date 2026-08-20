import networkx as nx
from typing import List, Dict, Any

def get_candidate_routes(G: nx.DiGraph, origin_node, dest_node, max_alternatives=4) -> List[Dict[str, Any]]:
    """
    Generates distinct, high-quality candidate routes:
    1. Direct Fastest (⚡ Fastest)
    2. CoolPath Recommended (❄️ Coolest)
    3. Balanced Route (⚖️ Balanced)
    4. Shaded Corridor (🌳 Shaded Side-Street)
    Computes avg_temp_c, travel time, and thermal metrics for each route.
    """
    try:
        fastest_nodes = nx.shortest_path(G, origin_node, dest_node, weight="travel_time")
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        try:
            fastest_nodes = nx.shortest_path(G, origin_node, dest_node, weight="walk_time")
        except Exception:
            return []

    def get_route_metrics(nodes):
        total_time = 0.0
        total_thermal = 0.0
        temps = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            edge = G[u][v]
            t = float(edge.get("travel_time", edge.get("walk_time", 1.0)))
            tc = float(edge.get("thermal_cost", 0.0))
            total_time += t
            total_thermal += tc
            if "temperature" in edge:
                try:
                    temps.append(float(edge["temperature"]))
                except Exception:
                    pass
        avg_temp = round(sum(temps) / len(temps), 1) if temps else 32.0
        return total_time, total_thermal, avg_temp

    routes = []
    seen_paths = set()

    def path_key(path):
        return tuple(path)

    # 1. Fastest Route (Baseline)
    f_time, f_thermal, f_avg_temp = get_route_metrics(fastest_nodes)
    fastest_route = {
        "id": "fastest",
        "name": "Direct Fastest",
        "tag": "⚡ Fastest",
        "nodes": fastest_nodes,
        "geometry": [[G.nodes[n]['x'], G.nodes[n]['y']] for n in fastest_nodes],
        "travel_time": f_time,
        "walk_time": f_time, # backward-compatibility
        "thermal_cost": f_thermal,
        "avg_temp_c": f_avg_temp,
        "is_fastest": True,
        "explanation": f"Direct route minimizing overall travel time. Average street temperature: {f_avg_temp}°C."
    }
    routes.append(fastest_route)
    seen_paths.add(path_key(fastest_nodes))

    max_allowed_time = f_time * 1.30 # Allow up to +30% travel time for cooler routes

    # 2. Pure Thermal Shortest Path (Coolest possible)
    try:
        coolest_nodes = nx.shortest_path(G, origin_node, dest_node, weight="thermal_cost")
        if path_key(coolest_nodes) not in seen_paths:
            c_time, c_thermal, c_avg_temp = get_route_metrics(coolest_nodes)
            if c_time <= max_allowed_time:
                routes.append({
                    "id": "coolest",
                    "name": "CoolPath Recommended",
                    "tag": "❄️ Coolest",
                    "nodes": coolest_nodes,
                    "geometry": [[G.nodes[n]['x'], G.nodes[n]['y']] for n in coolest_nodes],
                    "travel_time": c_time,
                    "walk_time": c_time,
                    "thermal_cost": c_thermal,
                    "avg_temp_c": c_avg_temp,
                    "is_fastest": False,
                    "explanation": f"Optimized for maximum heat avoidance. Follows cooler street microclimates at {c_avg_temp}°C."
                })
                seen_paths.add(path_key(coolest_nodes))
    except Exception:
        pass

    # 3. Multi-Objective Pareto Weights (Alpha Blending Time & Heat)
    for alpha in [0.5, 0.75, 0.3]:
        if len(routes) >= max_alternatives:
            break
        try:
            def blend_cost(u, v, d):
                wt = d.get("travel_time", d.get("walk_time", 1.0))
                tc = d.get("thermal_cost", 0.0)
                return (1.0 - alpha) * wt + alpha * tc

            path = nx.shortest_path(G, origin_node, dest_node, weight=blend_cost)
            pk = path_key(path)
            if pk not in seen_paths:
                p_time, p_thermal, p_avg_temp = get_route_metrics(path)
                if p_time <= max_allowed_time:
                    tag = "⚖️ Balanced" if alpha == 0.5 else "🌿 Shaded Option"
                    name = "Balanced Route" if alpha == 0.5 else "Shaded Alternative"
                    routes.append({
                        "id": f"route_{len(routes)}",
                        "name": name,
                        "tag": tag,
                        "nodes": path,
                        "geometry": [[G.nodes[n]['x'], G.nodes[n]['y']] for n in path],
                        "travel_time": p_time,
                        "walk_time": p_time,
                        "thermal_cost": p_thermal,
                        "avg_temp_c": p_avg_temp,
                        "is_fastest": False,
                        "explanation": f"Balanced compromise between speed and temperature ({p_avg_temp}°C)."
                    })
                    seen_paths.add(pk)
        except Exception:
            continue

    # 4. Corridor Diversification (Side-Street Detour away from main avenues)
    if len(routes) < max_alternatives and routes:
        used_edges = set()
        for r in routes:
            for u, v in zip(r["nodes"][:-1], r["nodes"][1:]):
                used_edges.add((u, v))
                
        try:
            def penalty_cost(u, v, d):
                base_cost = d.get("thermal_cost", 0.0) + d.get("travel_time", d.get("walk_time", 1.0)) * 0.5
                if (u, v) in used_edges:
                    return base_cost * 2.5
                return base_cost

            pen_path = nx.shortest_path(G, origin_node, dest_node, weight=penalty_cost)
            pk = path_key(pen_path)
            if pk not in seen_paths:
                pen_time, pen_thermal, pen_avg_temp = get_route_metrics(pen_path)
                if pen_time <= max_allowed_time:
                    routes.append({
                        "id": f"route_{len(routes)}",
                        "name": "Side-Street Corridor",
                        "tag": "🌳 Quiet Corridor",
                        "nodes": pen_path,
                        "geometry": [[G.nodes[n]['x'], G.nodes[n]['y']] for n in pen_path],
                        "travel_time": pen_time,
                        "walk_time": pen_time,
                        "thermal_cost": pen_thermal,
                        "avg_temp_c": pen_avg_temp,
                        "is_fastest": False,
                        "explanation": f"Avoids busy asphalt corridors via quieter side streets. Average temperature: {pen_avg_temp}°C."
                    })
                    seen_paths.add(pk)
        except Exception:
            pass

    return routes
