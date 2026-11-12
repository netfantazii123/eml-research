"""
distill_compare — classical distillation baselines to put EML results in context.

The thesis claim is that a neural policy can be distilled into a compact EML
formula. That claim is only meaningful next to what the established methods do
with the same oracle, the same data and the same evaluation seeds. This package
provides those methods behind one interface, the benchmark tasks to run them on,
and the runners that write comparable CSV rows.

Students (students.py), all fit on the TEACHER's outputs, never on ground truth:

    tree      CART decision tree on the teacher's argmax (behavioural cloning)
    viper     VIPER (Bastani et al., 2018): DAgger + loss-weighted CART (viper.py)
    linear    logistic / ridge regression on the teacher's outputs
    kd_mlp    small MLP student, Hinton et al. (2015) soft-target KD
              (classification) / Ba & Caruana (2014) logit matching (ranking)
    gplearn   genetic-programming symbolic regression (Koza-style, gplearn)
    eml       EML formula, DATA phase only, through emlcore's GA
    classic   the same GA over the control basis (+ - * / exp log tanh)

Task kinds:

    classify  X -> one of K classes / actions; teacher gives K logits per row
    rank      a group of candidates per decision, teacher gives one score per
              candidate, decision = argmax within the group (Tetris placements)
"""
