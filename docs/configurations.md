# Canonical policy configuration

The previous collection of actor/critic ablation configs was removed. Policy training now has one canonical configuration:

`configs/capacity_local.yaml`

Its purpose is to isolate the research question of interest: can a shared local recurrent rule construct globally strong bridges when global mechanics is available only as a training signal?

## Main choices

| Setting | Choice | Reason |
| --- | --- | --- |
| actor | shared GRU | same local rule on every robot, with memory |
| actor inputs | local observation/history only | decentralized execution |
| critic | graph critic | training-only partial-construction value estimate |
| episode end | fixed horizon | allows strengthening after the first span |
| structural reward | terminal capacity teacher | optimize physical bridge capacity, not graph proxy |
| individual credit | leave-one-robot-out capacity difference | structural credit assignment |
| default teacher | exact 6-DOF statics | ground truth |
| optional teacher | geometry-only CapacityGNN | faster large-scale RL rollouts |

Legacy model classes are still present for old checkpoint compatibility, but there are no separate `baseline`, `mlp`, `bio`, `FiLM`, traffic, heterogeneity, or curriculum YAML variants anymore.
