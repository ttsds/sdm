"""Light test: the CLI module imports and parses without the upstream `ttsds`
package being installed. The full pipeline requires the `teachers` extra.
"""

import importlib


def test_module_importable():
    mod = importlib.import_module("sdm.eval.correlate_with_mos")
    assert hasattr(mod, "main")


def test_load_specs_round_trip(tmp_path):
    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text(
        """
heads:
  - {name: hubert, target_dim: 768, pooled: false}
  - {name: whisper, target_dim: 768, pooled: true}
"""
    )
    from sdm.eval.correlate_with_mos import _load_specs

    specs = _load_specs(yaml_path)
    assert len(specs) == 2
    assert specs[0].name == "hubert"
    assert specs[1].pooled is True
