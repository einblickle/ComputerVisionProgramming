
fig = make_subplots(1, 2)
fig.add_trace(go.Image(z=image_baboon_red, colorscale = 'greys'), 1, 1)

fig.update_layout(height=600)
fig.show()
