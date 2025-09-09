from jinja2 import Template

# Create a simple template string
with open("ledro_d_fc.cir") as f:
    data = f.read()


# Create a Template object
template = Template(data)

state = dict({
    "nA1": 7.45e-08,
    "nB1": 6,
    "nA2": 1.4e-07,
    "nB2": 2,
    "nA3": 3.75e-08,
    "nB3": 3,
    "nA4": 3.04e-07,
    "nB4": 3,
    "nA5": 3.72e-08,
    "nB5": 4,
    "nA6": 1.24e-07,
    "nB6": 2,
    "vbiasp1": 0.659,
    "vbiasp2": 0.408,
    "vbiasn0": 0.0525,
    "vbiasn1": 0.016,
    "vbiasn2": 0.352,
    "vcm": 0.4,
    "vdd": 0.8,
    "tempc": 27,

    "design_path": "/tmp"
})
# Render the template with a variable
output = template.render(state)

print(output)