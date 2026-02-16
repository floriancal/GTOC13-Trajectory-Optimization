import numpy as np 
import pykep as pk
import plotly.graph_objects as go

def planet_orbit_plotly(planet, N=400):
    ts = np.linspace(0, planet.compute_period(pk.epoch(0, 'mjd2000')), N)

    xs, ys, zs = [], [], []

    for t in ts:
        r, _ = planet.eph(t / 86400.0)  # PyKEP attend des jours
        xs.append(float(r[0]))
        ys.append(float(r[1]))
        zs.append(float(r[2]))
    return xs, ys, zs
 
def follow_camera(xs, ys, zs, dist):
    return dict(
        eye=dict(x=dist, y=dist, z=dist),
        center = dict(x=1, y=1, z=1)
    )

def lim(arr, margin=0.2):
    m = max(arr) - min(arr)
    return [min(arr) - margin*m, max(arr) + margin*m]


def build_html_3D_anim(traj_plot, planets):

    # Buid layout with 4 buttons 
    layout = go.Layout(
        title="Start Title",
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(
                        label="Slow",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=300, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    ),
                    dict(
                        label="Normal",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    ),
                    dict(
                        label="Fast",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=1, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate"
                        )]
                    )
                ]
            )
        ]
    )

    # Vectors for s/c positions
    x_sc, y_sc, z_sc = [], [], []

    # Accumulateurs globaux (SC + planètes + orbites)
    xs_all, ys_all, zs_all = [], [], []

    for arc, t, r in traj_plot:
        x = float(r[0])
        y = float(r[1])
        z = float(r[2])

        x_sc.append(x)
        y_sc.append(y)
        z_sc.append(z)


    max_range = max(
    max(x_sc) - min(x_sc),
    max(y_sc) - min(y_sc),
    max(z_sc) - min(z_sc)
    )


    # List for the frames
    frames = []
    for i, (arc, t, r) in enumerate(traj_plot):
        
        # modifiable camera angle (3 three args defines the eye position 4th defines the distance)
        camera = follow_camera(1, 1, 1, 1 )

        # add the planets positions
        planet_traces = []
        for planet in planets.values():
            r_p, _ = planet.eph(t / 86400.0)
            planet_traces.append(go.Scatter3d(
                x=[float(r_p[0])],
                y=[float(r_p[1])],                 
                z=[float(r_p[2])],
                mode="markers",
                marker=dict(size=6),
                showlegend=False
            ))
        
        # build the frame with s/c position (stacing the positions)
        frames.append(go.Frame(
            data=[
                *planet_traces,
                 go.Scatter3d(
                    x=x_sc[:i+1],
                    y=y_sc[:i+1],
                    z=z_sc[:i+1],
                    mode="lines",
                    line=dict(width=4, color="red"),
                    )],
            layout=dict(
                scene=dict(
                camera=camera
            ),
            ),
            ))
        
    # Planets intial pos
    initial_planet_pos = []
    for planet in planets.values():
        r_p, _ = planet.eph(traj_plot[0][1] / 86400.0)
        initial_planet_pos.append(go.Scatter3d(
            x=[float(r_p[0])],
            y=[float(r_p[1])],
            z=[float(r_p[2])],
            mode="markers",
            marker=dict(size=6),
            showlegend=False
        ))

    # Initial figure (frame 0)
    fig = go.Figure(
    data=[*initial_planet_pos],
    frames=frames,
    layout=layout
    )
    
    
    fig.add_trace(go.Scatter3d(
    x=[x_sc[0]],
    y=[y_sc[0]],
    z=[z_sc[0]],
    mode="lines",
    line=dict(width=4, color="red"),
    name="S/C",
    showlegend=True
    ))

    # Planets orbits
    for planet in planets.values():
        xp, yp, zp = planet_orbit_plotly(planet)

        fig.add_trace(go.Scatter3d(
                x=xp,
                y=yp,
                z=zp,
                mode="lines",
                line=dict(width=1),
                name=f"{planet.name} orbit",
                showlegend=True
            ))
    



    fig.write_html("3D_sim.html", include_plotlyjs="cdn", auto_play = False)