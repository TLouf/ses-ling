import matplotlib.pyplot as plt


def variant_evol(evol_df, nr_agents_per_class: dict, **sim_kwargs):
    fig, ax = plt.subplots()
    x_plot = evol_df.index
    for col in evol_df.columns:
        ax.plot(x_plot, nr_agents_per_class[col] - evol_df[col], label=col+1)
    ax.legend(title='SES class')
    ax.set_xlabel('time')
    ax.set_ylabel('agents using standard form')
    ax.set_title(", ".join(f"{key} = {value}" for key, value in sim_kwargs.items()))
    return ax
