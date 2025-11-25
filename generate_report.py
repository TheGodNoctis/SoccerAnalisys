import numpy as np

def generate_report(tracks, passes, pass_stats, team_ball_control):
    report = []
    report.append("==== RELATÓRIO DE ESTATÍSTICAS ====\n")

    # 3. Posse de bola
    team_1_time = np.sum(team_ball_control == 1)
    team_2_time = np.sum(team_ball_control == 2)
    total_time = len(team_ball_control)
    team_1_perc = (team_1_time / total_time) * 100 if total_time > 0 else 0
    team_2_perc = (team_2_time / total_time) * 100 if total_time > 0 else 0
    report.append("\n--- POSSE DE BOLA ---")
    report.append(f"\nTime 1: {team_1_perc:.1f}%")
    report.append(f"\nTime 2: {team_2_perc:.1f}%\n")

    # 5. Estatísticas de passes
    report.append("\n\n--- ESTATÍSTICAS DE PASSES ---")
    for team, s in pass_stats.items():
        report.append(
            f"\nTime {team}: Total={s['total']}, Certos={s['certos']}, Errados={s['total'] - s['certos']}, "
            f"Precisão={s['percentual_acerto']}%, Interceptados={s['interceptados']}"
        )

    # 6. Distância percorrida e velocidades
    distances = {}
    speeds_instant = {}
    speeds_avg = {}
    
    for frame in tracks["players"]:
        for pid, p_info in frame.items():
            # Distância acumulada (em metros no track)
            distances[pid] = distances.get(pid, 0) + p_info.get("distance", 0)
            
            # Velocidade instantânea
            speed_val = p_info.get("speed", 0)
            if pid not in speeds_instant or speed_val > speeds_instant[pid]:
                speeds_instant[pid] = speed_val
            
            # Velocidade média
            speeds_avg.setdefault(pid, []).append(speed_val)

    report.append("\n\n--- MAIORES DISTÂNCIAS PERCORRIDAS (m) ---")
    for pid, dist_m in sorted(distances.items(), key=lambda x: x[1], reverse=True)[:5]:
        dist_km = dist_m / 100
        report.append(f"\nJogador {pid}: {dist_km:.2f} m")

    report.append("\n\n--- MAIORES VELOCIDADES ALCANÇADAS (m/s) ---")
    for pid, speed in sorted(speeds_instant.items(), key=lambda x: x[1], reverse=True)[:5]:
        report.append(f"\nJogador {pid}: {speed:.2f} m/s")

    avg_speeds = {pid: np.mean(vals) for pid, vals in speeds_avg.items()}
    report.append("\n\n--- MAIORES VELOCIDADES MÉDIAS (m/s) ---")
    for pid, avg_speed in sorted(avg_speeds.items(), key=lambda x: x[1], reverse=True)[:5]:
        report.append(f"\nJogador {pid}: {avg_speed:.2f} m/s")

    report.append("\n\n==== FIM DO RELATÓRIO ====")
    return "\n".join(report)
