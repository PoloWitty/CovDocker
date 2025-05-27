
### Unseen Test Set Evaluation Scripts
The scripts in this folder are early-stage implementations developed for evaluating performance on unseen test sets. These scripts have **not** been tested against the latest codebase and were excluded from the final paper. However, they may serve as a useful starting point for further experimentation. So we have released it for others to refer to.

For details on the unseen dataset preprocessing pipeline, refer to [`scripts/preprocess.sh`](scripts/preprocess.sh).

---

### Dataset Description
Following [EquiBind](https://github.com/HannesStark/EquiBind), we constructed an unseen test set by filtering samples with **sequence similarity < 0.8** relative to the training set.

#### Key Observations from Preliminary Experiments:
1. **Performance Drop on Unseen Data**:
   - The method described in the paper exhibits a notable decline in performance (especially in **Task 1**) when evaluated on the unseen test set, whereas traditional methods show relatively smaller declines
   - This trend aligns with findings in [PoseBuster](https://github.com/maabuu/posebusters), where deep learning methods generally show larger performance drops on unseen data compared to traditional approaches.

2. **Implications for Robustness**:
   - The results suggest that current deep learning models still lack robustness when generalizing to unseen samples, highlighting an area for future improvement.
