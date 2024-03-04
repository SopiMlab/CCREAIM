import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import torch
import mplcursors

from utils import return_clip_from_path

SOURCE_LABEL = "source"
TARGET_LABEL = "target"

LINE_WIDTH_SCALE = 3

live = True

using_hover_attention = False

if using_hover_attention:
    hover_target = None

if live:
    attn_weights = torch.load("attention_data/live_attention.pt")
    probs = torch.load("attention_data/live_probs.pt")
else:  # NOTE: Currently not used! This is data from running test_model_bank.ipynb
    gen = 1
    start_sec = 0
    save_path_root = "{}/../notebooks/outputs".format(os.getcwd())
    test_fn = "piano_man"
    attn_weights = torch.load('{}/{}/{}-gen{}-sec{}-attn_weights.pt'.format(save_path_root, test_fn, test_fn, gen, start_sec))[0]
    wav_fn = '{}/{}/{}-gen{}-sec{}.wav'.format(save_path_root, test_fn, test_fn, gen, start_sec)

input_length = probs.size(1)
total_length = attn_weights.size(0)
num_heads = attn_weights.size(1)
num_heads_x = [(k + 1) / (num_heads + 1) - 0.5 for k in range(num_heads)]
print(attn_weights.shape, probs.shape)
print(input_length, total_length, num_heads)

# For better plots
sns.set()

head_colors = ["y-", "b-", "g-", "r-"]

bars = []
animation_bars = []

colors =  {
    SOURCE_LABEL: ["lightcoral", "indianred"],
    TARGET_LABEL: ["skyblue", "deepskyblue"],
}

to_other =  {
    SOURCE_LABEL: TARGET_LABEL,
    TARGET_LABEL: SOURCE_LABEL,
}

def cumulative_combine(lists):
    result = []
    temp = []
    for sublist in lists:
        temp += sublist  # Combine the current sublist with the accumulated list
        result.append(temp.copy())  # Append a copy of the accumulated list to result
    return result

def get_center_coords_of_bar(b):
    w,h = b.get_width(), b.get_height()
    # lower left vertex
    x, y = b.xy
    return x + w / 2, y + h / 2

def get_relevant_bars_from_other(label, lower, upper):
    output = []
    for bar in bars:
        l, s = bar.get_label().split("_")
        if f"{to_other[label]}" in l and lower <= int(s[3:]) < upper:
            output.append(bar)
    return sorted(output, key=lambda b: int(b.get_label().split("_")[1][3:]))

def plot_attention(attn_weights, probs):
    fig, ax = plt.subplots(figsize=(10, 3), sharex=True)
    ax.title.set_text(f"Relative attention visualization")
    ax.invert_yaxis()
    ax.set_xlabel("seconds")

    for i in range(total_length):
        sec_bars = []
        lines = []
        # source
        if i < total_length-1:
            b = ax.barh(SOURCE_LABEL, 1, left=i, height=0.2, 
                        label=f"{SOURCE_LABEL}_sec{i}", color=colors[SOURCE_LABEL][0])
            bars.append(b)
            sec_bars.append(b.patches[0])

        # target
        if i >= input_length:
            b = ax.barh(TARGET_LABEL, 1, left=i, height=0.2, 
                        label=f"{TARGET_LABEL}_sec{i}", color=colors[TARGET_LABEL][0])
            relevant_bars = get_relevant_bars_from_other(TARGET_LABEL, i-input_length, i)
            p = probs[i]
            for j,other_bar in enumerate(relevant_bars):
                x0,y0 = get_center_coords_of_bar(b.patches[0])
                x1,y1 = get_center_coords_of_bar(other_bar.patches[0])
                aw = attn_weights[i]
                for head in range(num_heads):
                    src_tok_total_attn = aw[head,:,j]
                    if src_tok_total_attn[j] >= 0.1:
                        l = ax.plot([x0 + num_heads_x[head], x1 + num_heads_x[head]], [y0, y1], head_colors[head], label=f"{SOURCE_LABEL}_sec{i-input_length+j}_{TARGET_LABEL}_sec{i}_head{head}", linewidth=LINE_WIDTH_SCALE * src_tok_total_attn.sum(), visible=not using_hover_attention)
                        lines.append(l[0])
            bars.append(b)
            sec_bars.append(b.patches[0])
        print(sec_bars, lines, sec_bars + lines)
        animation_bars.append(sec_bars + lines)


    # Define custom legend
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='yellow', lw=4, label='Head 1'),
        matplotlib.lines.Line2D([0], [0], color='blue', lw=4, label='Head 2'),
        matplotlib.lines.Line2D([0], [0], color='green', lw=4, label='Head 3'),
        matplotlib.lines.Line2D([0], [0], color='red', lw=4, label='Head 4')
    ]

    # Add the custom legend to the plot
    plt.legend(handles=legend_elements, loc='lower left')

    # for i in range(num_heads):
            # plt.matshow(attn_weights[i].detach().cpu().numpy(), cmap='viridis', fignum=False)
            # ax.title(f'Head {i}')

    ax.set_ylim(1.5, -0.5)
    ax.set_xlim(0, total_length)

    return fig, ax

fig, ax = plot_attention(attn_weights=attn_weights, probs=probs)

def get_relevant_lines(label, lower, upper):
    idx = 1 if SOURCE_LABEL in label else 3
    global hover_target
    output = []
    if hover_target is None or label != hover_target:
        hover_target = label
        for line in ax.lines:
            sec = int(line.get_label().split("_")[idx][3:])
            print(line.get_label(), lower, sec, upper)
            if lower <= sec < upper:
                output.append(line)
    return output

def on_hover_color(sel):
    if type(sel.artist) == matplotlib.lines.Line2D:
        return
    bar = sel.artist.patches[0]
    # Change the color of the selected bar
    label = sel.artist.get_label().split("_")[0]
    bar.set_color(colors[label][1])
    bar.set_edgecolor("white")
    # Deselect and revert color upon moving away
    def on_unhover_color(event):
        if bar.contains(event)[0]:
            return
        bar.set_color(colors[label][0])
        bar.set_edgecolor("white")
        fig.canvas.draw_idle()
        fig.canvas.mpl_disconnect(disconnect_id)
    disconnect_id = fig.canvas.mpl_connect('motion_notify_event', on_unhover_color)

def on_hover_annotation(sel):
    if type(sel.artist) == matplotlib.lines.Line2D:
        sel.annotation.set_text(sel.artist.get_lw() / LINE_WIDTH_SCALE)
        return
    sel.annotation.set_text(sel.artist.get_label())
    def on_unhover_annotation(event):
        fig.canvas.draw_idle()
        fig.canvas.mpl_disconnect(disconnect_id)
    disconnect_id = fig.canvas.mpl_connect('motion_notify_event', on_unhover_annotation)

def on_hover_attention(sel):
    if type(sel.artist) == matplotlib.lines.Line2D:
        return
    bar = sel.artist.patches[0]
    # Show the attentions from the selected token
    label, sec = sel.artist.get_label().split("_")
    sec = int(sec[3:])
    if label == TARGET_LABEL:
        relevant_lines = get_relevant_lines(sel.artist.get_label(), sec - input_length + 1, sec + 1)
    else:
        relevant_lines = get_relevant_lines(sel.artist.get_label(), sec + 1, sec + 1 + input_length)
    for line in relevant_lines:
        line.set_visible(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

    def on_unhover_attention(event):
        for line in relevant_lines:
            line.set_visible(False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        fig.canvas.mpl_disconnect(disconnect_id)
    disconnect_id = fig.canvas.mpl_connect('motion_notify_event', on_unhover_attention)
    

# def on_hover_attention(sel):
#     bar = sel.artist.patches[0]
#     # Show the attentions from the selected token
#     label, head, sec = sel.artist.get_label().split("_")
#     head, sec = int(head[4:]), int(sec[3:])
#     print(label, head, sec)
#     if label == TARGET_LABEL:
#         attention_data = attn_weights[head,sec-input_length,:]
#     else:
#         attention_data = attn_weights[head,:,sec]
#     print(attention_data)
#     relevant_bars = get_relevant_bars_from_other(label, head)
#     for other_bar in relevant_bars:
#         x0,y0 = get_center_coords_of_bar(bar)
#         x1,y1 = get_center_coords_of_bar(other_bar.patches[0])
#         s = int(other_bar.get_label().split("_")[2][3:])
#         ax[head].plot([x0, x1], [y0, y1], head_colors[head], linewidth=attention_data[s]*5)
#     # Deselect and revert color upon moving away
#     def on_unhover_attention(event):
#         for line in ax[head].get_lines(): # ax.lines:
#             line.remove()
#         # fig.canvas.draw_idle()
#         fig.canvas.mpl_disconnect(disconnect_id)
#     disconnect_id = fig.canvas.mpl_connect('motion_notify_event', on_unhover_attention)


cursor = mplcursors.cursor(hover=True)
cursor.connect("add", on_hover_color)
if using_hover_attention:
    cursor.connect("add", on_hover_attention)
cursor.connect("add", on_hover_annotation)


def concat_audio_files(file1_path, file2_path):
    from scipy.io.wavfile import write
    import numpy as np
    sample_rate = 22050
    tensor1 = torch.load(file1_path)  # Replace with your file paths
    tensor2 = torch.load(file2_path)
    tensor1 = torch.cat((tensor1, torch.Tensor(sample_rate, 1)), dim=0)
    if tensor1.shape[1] != tensor2.shape[1]:
        print("Error: The tensors have a different number of seconds. Please make sure they are compatible.")
    merged_tensor = torch.stack([tensor1.squeeze(1), tensor2.squeeze(1)])
    print(merged_tensor.shape)
    merged_numpy = merged_tensor.numpy()

    # Convert to int16 format because scipy.io.wavfile expects PCM format
    # You might lose some information due to this conversion. Be sure to normalize your tensors to be within [-1, 1] before converting.
    merged_numpy_int16 = np.int16(merged_numpy * 32767)

    # Write to a WAV file
    write('attention_data/live_model.wav', sample_rate, merged_numpy_int16.T)  # 22050 is the sample rate

def attention_visualization_video(video_path, audio_path, output_path):
    import ffmpeg
    video  = ffmpeg.input(video_path).video # get only video channel
    audio  = ffmpeg.input(audio_path).audio # get only audio channel
    output = ffmpeg.output(video, audio, output_path, vcodec='copy', acodec='aac', strict='experimental')
    ffmpeg.run(output, overwrite_output=True)

print(bars)
print("")
print(animation_bars)

concat_audio_files('attention_data/live_model_input.pt', 'attention_data/live_model_output.pt')
ani = animation.ArtistAnimation(fig=fig, artists=cumulative_combine(animation_bars), interval=1000)
ani.save("attention_data/live_attention_visualization.mp4", animation.FFMpegWriter(fps=1))
attention_visualization_video("attention_data/live_attention_visualization.mp4", "attention_data/live_model.wav", "attention_data/final_live_attention_visualization.mp4")
plt.show()