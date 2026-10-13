import pytest

from primagate.core import steps
from primagate.drivers.sync import SyncDriver
from primagate.schema.errors import PrimaGateError


def test_every_closed_set_action_has_a_handler() -> None:
    missing = set(steps.ACTION_TYPES) - set(SyncDriver.HANDLERS)
    assert not missing, missing


def test_driver_raises_on_unhandled_action() -> None:
    class Unknown:
        pass

    driver = SyncDriver(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    with pytest.raises(PrimaGateError, match="unhandled action"):
        driver.perform(Unknown())  # type: ignore[arg-type]
