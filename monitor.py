from backtester import Backtester
from config import Config

class Monitor:
    def __init__(self, config, trade_obj: Backtester) -> None:

        self.trade_obj = trade_obj
        self.order_status = self.trade_obj.order_status

        self.config = config
        self.open_positions = {}

    def add_position(self, contract):
        self.open_positions[contract['id']] = contract

    async def check_open_positions(self, api):
        for contract_id, contract in list(self.open_positions.items()):
            try:
                updated_contract = await api.proposal_open_contract(contract_id)
                if updated_contract['status'] == 'closed':
                    del self.open_positions[contract_id]
                elif self.should_apply_trailing_stop(updated_contract):
                    await self.apply_trailing_stop(api, updated_contract)
            except Exception as e:
                print(f"Error checking position {contract_id}: {e}")

    def should_apply_trailing_stop(self, contract):
        if 'current_spot' not in contract or 'entry_spot' not in contract:
            return False
        profit_pips = (contract['current_spot'] - contract['entry_spot']) * 10000
        return profit_pips > self.config.TRAILING_STOP_PIPS

    async def apply_trailing_stop(self, api, contract):
        new_stop_loss = contract['current_spot'] - (self.config.TRAILING_STOP_PIPS / 10000)
        try:
            await api.sell({
                "contract_id": contract['id'],
                "price": new_stop_loss
            })
        except Exception as e:
            print(f"Error applying trailing stop to {contract['id']}: {e}")


    @property
    def is_cancelled(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was filled or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `FILLED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'FILLED':
            return True
        else:
            return False

    @property
    def is_rejected(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was rejected or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `REJECTED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'REJECTED':
            return True
        else:
            return False

    @property
    def is_expired(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order has expired or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `EXPIRED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'EXPIRED':
            return True
        else:
            return False

    @property
    def is_replaced(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order has been replaced or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `REPLACED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'REPLACED':
            return True
        else:
            return False

    @property
    def is_working(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is working or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `WORKING`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'WORKING':
            return True
        else:
            return False

    @property
    def is_pending_activation(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending activation or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `PENDING_ACTIVATION`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_ACTIVATION':
            return True
        else:
            return False

    @property
    def is_pending_cancel(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending cancellation or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `PENDING_CANCEL`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_CANCEL':
            return True
        else:
            return False

    @property
    def is_pending_replace(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending replacement or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `PENDING_REPLACE`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_REPLACE':
            return True
        else:
            return False

    @property
    def is_queued(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is in the queue or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `QUEUED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'QUEUED':
            return True
        else:
            return False

    @property
    def is_accepted(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was accepted or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `ACCEPTED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'ACCEPTED':
            return True
        else:
            return False

    @property
    def is_awaiting_parent_order(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is waiting for the parent order
        to execute or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `AWAITING_PARENT_ORDER`,
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'AWAITING_PARENT_ORDER':
            return True
        else:
            return False

    @property
    def is_awaiting_condition(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is waiting for the condition
        to execute or not.

        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.

        Returns
        -------
        bool
            `True` if the order status is `AWAITING_CONDITION`,
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'AWAITING_CONDITION':
            return True
        else:
            return False
