from kinova_lfd.envs.base.mujoco.configclass import configclass


@configclass
class ItemCfg:
    value = 1


@configclass
class ContainerCfg:
    items = (ItemCfg(),)


def test_from_dict_preserves_tuple_config_objects() -> None:
    cfg = ContainerCfg()

    cfg.from_dict({"items": [{"value": 2}]})

    assert isinstance(cfg.items[0], ItemCfg)
    assert cfg.items[0].value == 2


def test_from_dict_keeps_tuple_of_dicts_as_dicts() -> None:
    cfg = {"items": ({"value": 1},)}

    # This exercises the existing dictionary-container behavior.
    from kinova_lfd.envs.base.mujoco.configclass import update_class_from_dict

    update_class_from_dict(cfg, {"items": [{"value": 2}]})

    assert cfg["items"] == ({"value": 2},)
