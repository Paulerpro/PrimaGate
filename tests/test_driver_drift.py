import pytest

from primagate.core import steps
from primagate.drivers.async_ import AsyncDriver
from primagate.drivers.sync import SyncDriver
from primagate.schema.errors import PrimaGateError


@pytest.mark.parametrize("driver_cls", [SyncDriver, AsyncDriver])
def test_every_closed_set_action_has_a_handler(driver_cls: type) -> None:
    missing = set(steps.ACTION_TYPES) - set(driver_cls.HANDLERS)
    assert not missing, missing


def test_both_drivers_handle_the_same_actions() -> None:
    assert set(AsyncDriver.HANDLERS) == set(SyncDriver.HANDLERS)


def test_driver_raises_on_unhandled_action() -> None:
    class Unknown:
        pass

    driver = SyncDriver(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    with pytest.raises(PrimaGateError, match="unhandled action"):
        driver.perform(Unknown())  # type: ignore[arg-type]


async def test_async_driver_raises_on_unhandled_action() -> None:
    class Unknown:
        pass

    driver = AsyncDriver(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    with pytest.raises(PrimaGateError, match="unhandled action"):
        await driver.perform(Unknown())  # type: ignore[arg-type]
