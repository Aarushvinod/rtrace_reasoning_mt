"""
fasttext_compat.py
──────────────────
Numpy-2.x compatibility shim for the fasttext package. The upstream
`_FastText.predict` binding predates NumPy's 1.25 array-scalar tightening; on
NumPy 2.x it can raise on `np.asarray(zip(...))`-style paths. Calling
`_patch_fasttext_for_numpy2()` swaps in a version that copes with both single
strings and lists of strings and returns a plain ndarray of probabilities.

Inputs:  none (patches the imported `fasttext._FastText` class in place).
Outputs: side-effect only; after the call `_FastText.predict` is patched.
"""

import numpy as np


def _patch_fasttext_for_numpy2():
    try:
        from fasttext.FastText import _FastText
    except Exception:
        return

    def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        def _check(entry):
            if entry.find("\n") != -1:
                raise ValueError("predict processes one line at a time (remove '\\n')")
            return entry + "\n"

        if isinstance(text, list):
            text = [_check(t) for t in text]
            all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
            return all_labels, all_probs
        else:
            text = _check(text)
            predictions = self.f.predict(text, k, threshold, on_unicode_error)
            if predictions:
                probs, labels = zip(*predictions)
            else:
                probs, labels = ([], ())
            return labels, np.asarray(probs)

    _FastText.predict = _patched_predict
