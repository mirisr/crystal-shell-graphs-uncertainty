# Final NCA checkpoint manifest

The selected manuscript model is `resnet_crystal_edgegate_rd_k8`. The five released checkpoints were copied from the final Cradle run tree on October 10, 2026. Each checkpoint contains the trained state dictionary, feature normalization vectors, complete training configuration, selected epoch, and validation metrics.

| Fold | File | Bytes | SHA-256 |
| ---: | --- | ---: | --- |
| 0 | `checkpoints/fold0/best.pt` | 434110 | `bec94ac2ff91e655a782decb692f84926912cb25868e77b0e348dbcab759346f` |
| 1 | `checkpoints/fold1/best.pt` | 434110 | `0eaf8f4341b17eaa6c54640d472ecd2501ecb26026094d4d45411ff58f5b000e` |
| 2 | `checkpoints/fold2/best.pt` | 434110 | `4fdf664c0422cd4d8b4a67a9e4c3fcd193ac2166fcc68d5ecd0f5a51ab253e4d` |
| 3 | `checkpoints/fold3/best.pt` | 434110 | `0e3df2eb8a523c43e27e3361574eecaddbcb43da558d2d38bf88c261ae92ec7d` |
| 4 | `checkpoints/fold4/best.pt` | 434110 | `e72f3dea8c4414ebe9704100e0e6d721654466cfe01fe31d30f1a4619a75afd7` |

Verify all files from the repository root:

```bash
(cd models && shasum -a 256 -c checksums.sha256)
python scripts/smoke_test.py
```

The checkpoint configuration uses 518 input channels, including six structured channels and a 512-dimensional frozen ResNet-18 embedding. The model predicts updates to the six structured channels while preserving the image representation as context.

These checkpoints belong only to this manuscript release. They are not checkpoints from the earlier `breast-dce-mri-graph-rollouts` repository.
