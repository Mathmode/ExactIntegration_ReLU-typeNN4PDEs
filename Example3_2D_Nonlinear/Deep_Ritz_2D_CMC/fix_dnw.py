import re

with open("My_integration.py", "r") as f:
    content = f.read()

# Replace the loop in my_integration_DNW
old_loop = """        for j in tf.range(num_neurons):
            is_sampled = tf.reduce_any(tf.equal(sampled_indices_j, j))
            if is_sampled:
                line_coef_j = tf.cast(tf.stack([wi0[j], wi1[j], bi[j]]), tf.float64)"""

new_loop = """        for j in sampled_indices_j:
            line_coef_j = tf.cast(tf.stack([wi0[j], wi1[j], bi[j]]), tf.float64)"""

content = content.replace(old_loop, new_loop)

with open("My_integration.py", "w") as f:
    f.write(content)
