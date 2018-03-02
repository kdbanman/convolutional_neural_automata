from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.core.display import display, HTML

def inline_video(environment, frame_interval_millis = 50, frame_count = 100, iterations_per_frame = 1, loop = False):
    fig, ax = plt.subplots();
    mat = ax.matshow(environment.grid);

    def update(_data):
        for i in range(iterations_per_frame):
            environment.iterate()

        mat.set_data(environment.grid);
        return [mat]

    anim = animation.FuncAnimation(fig, update, interval=frame_interval_millis, frames=frame_count, repeat=loop);
    display(HTML(anim.to_html5_video()))
    
    # close the figure to avoid rendering it separately, since it was just used to build the video
    plt.close(fig)