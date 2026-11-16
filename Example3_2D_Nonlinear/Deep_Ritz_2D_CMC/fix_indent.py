import re

with open("My_integration.py", "r") as f:
    content = f.read()

bad_block = """        for j in sampled_indices_j:
            line_coef_j = tf.cast(tf.stack([wi0[j], wi1[j], bi[j]]), tf.float64)
                domain_ij = clip_polygon_with_line_diff(domain_i, line_coef_j)
                
                if tf.shape(domain_ij)[0] > 0:
                    K_ij = compute_MC_integrals_K_ij(domain_ij, pts, wts, wi0[i], wi1[i], bi[i], ci[i], wi0[j], wi1[j], bi[j], ci[j], alpha)
                    Integral_du_du += K_ij"""

good_block = """        for j in sampled_indices_j:
            line_coef_j = tf.cast(tf.stack([wi0[j], wi1[j], bi[j]]), tf.float64)
            domain_ij = clip_polygon_with_line_diff(domain_i, line_coef_j)
            
            if tf.shape(domain_ij)[0] > 0:
                K_ij = compute_MC_integrals_K_ij(domain_ij, pts, wts, wi0[i], wi1[i], bi[i], ci[i], wi0[j], wi1[j], bi[j], ci[j], alpha)
                Integral_du_du += K_ij"""

content = content.replace(bad_block, good_block)

with open("My_integration.py", "w") as f:
    f.write(content)
