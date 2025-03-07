import pandas as pd
from datetime import datetime
import os
import numpy as np
from deriv_api import DerivAPI

import json
from datetime import datetime

from typing import List
from typing import Dict

from backtesting import Backtest, Strategy


class Backtester:
    def __init__(self, config, api, data_manager, strategy_manager):
        """
        Object Type:
        ----
        `pyrobot.Trade`

        Overview:
        ----
        Represents the Trade Object which is used to create new trades,
        add customizations to them, and easily modify existing content.
        """

        """Initializes a new order."""
        self.order = {}
        self.trade_id = ""
        self.order_id = ""
        self.account = ""
        self.order_status = "NOT_PLACED"

        self.side = ""
        self.side_opposite = ""
        self.enter_or_exit = ""
        self.enter_or_exit_opposite = ""

        self._order_response = {}
        self._triggered_added = False
        self._multi_leg = False
        self._one_cancels_other = False

        self.config = config
        self.api = api
        self.strategy_manager = strategy_manager
        self.data_manager = data_manager  # Make sure to pass the DataManager instance

    def to_dict(self) -> dict:
        # Initialize the Dict.
        obj_dict = {
            "__class___": self.__class__.__name__,
            "__module___": self.__module__,
        }

        # Convert the strategy_manager and data_manager objects to dicts if possible.
        if hasattr(self.strategy_manager, "to_dict"):
            obj_dict["strategy_manager"] = self.strategy_manager.to_dict()
        else:
            obj_dict["strategy_manager"] = str(
                self.strategy_manager
            )  # Fallback if no to_dict() method

        if hasattr(self.data_manager, "to_dict"):
            obj_dict["data_manager"] = self.data_manager.to_dict()
        else:
            obj_dict["data_manager"] = str(
                self.data_manager
            )  # Fallback if no to_dict() method

        # Add the Object's attributes.
        obj_dict.update(self.__dict__)

        # Convert to JSON string with indentation
        formatted_dict = json.dumps(obj_dict, indent=4)

        # Modify indentation to include '@@' and 4 spaces
        formatted_dict_with_prefix = "\n".join(
            [f"@@    {line}" for line in formatted_dict.splitlines()]
        )

        return formatted_dict_with_prefix

    def new_trade(
        self,
        trade_id: str,
        order_type: str,
        side: str,
        enter_or_exit: str,
        price: float = 0.00,
        stop_limit_price: float = 0.00,
    ) -> dict:
        """Creates a new Trade object template.

        A trade object is a template that can be used to help build complex trades
        that normally are prone to errors when writing the JSON. Additionally, it
        will help the process of storing trades easier.

        Arguments:
        ----
        order_type {str} -- The type of order you would like to create. Can be
            one of the following: ['mkt', 'lmt', 'stop', 'stop_lmt', 'trailing_stop']

        side {str} -- The side the trade will take, can be one of the
            following: ['long', 'short']

        enter_or_exit {str} -- Specifices whether this trade will enter a new position
            or exit an existing position. If used to enter then specify, 'enter'. If
            used to exit a trade specify, 'exit'.

        Returns:
        ----
        {dict} -- [description]
        """

        self.trade_id = trade_id

        self.order_types = {
            "mkt": "MARKET",
            "lmt": "LIMIT",
            "stop": "STOP",
            "stop_lmt": "STOP_LIMIT",
            "trailing_stop": "TRAILING_STOP",
        }

        self.order_instructions = {
            "enter": {"long": "BUY", "short": "SELL_SHORT"},
            "exit": {"long": "SELL", "short": "BUY_TO_COVER"},
        }

        self.order = {
            "orderStrategyType": "SINGLE",
            "orderType": self.order_types[order_type],
            "session": "NORMAL",
            "duration": "DAY",
            "orderLegCollection": [
                {
                    "instruction": self.order_instructions[enter_or_exit][side],
                    "quantity": 0,
                    "instrument": {"symbol": None, "assetType": None},
                }
            ],
        }

        if self.order["orderType"] == "STOP":
            self.order["stopPrice"] = price

        elif self.order["orderType"] == "LIMIT":
            self.order["price"] = price

        elif self.order["orderType"] == "STOP_LIMIT":
            self.order["price"] = stop_limit_price
            self.order["stopPrice"] = price

        elif self.order["orderType"] == "TRAILING_STOP":
            self.order["stopPriceLinkBasis"] = ""
            self.order["stopPriceLinkType"] = ""
            self.order["stopPriceOffset"] = 0.00
            self.order["stopType"] = "STANDARD"

        # Make a refrence to the side we take, useful when adding other components.
        self.enter_or_exit = enter_or_exit
        self.side = side
        self.order_type = order_type
        self.price = price

        # If it's a stop limit order or stop order, set the stop price.
        if self.is_stop_order or self.is_stop_limit_order:
            self.stop_price = price
        else:
            self.stop_price = 0.0

        # If it's a stop limit order set the stop limit price.
        if self.is_stop_limit_order:
            self.stop_limit_price = stop_limit_price
        else:
            self.stop_limit_price = 0.0

        # If it's a limit price set the limit price.
        if self.is_limit_order:
            self.limit_price = price
        else:
            self.limit_price = 0.0

        # Set the enter or exit state.
        if self.enter_or_exit == "enter":
            self.enter_or_exit_opposite = "exit"
        if self.enter_or_exit == "exit":
            self.enter_or_exit_opposite = "enter"

        # Set the side state.
        if self.side == "long":
            self.side_opposite = "short"
        if self.side == "short":
            self.side_opposite = "long"

        return self.order

    def instrument(
        self,
        symbol: str,
        quantity: int,
        asset_type: str,
        sub_asset_type: str = None,
        order_leg_id: int = 0,
    ) -> dict:
        """Adds an instrument to a trade.

        Arguments:
        ----
        symbol {str} -- The instrument ticker symbol.

        quantity {int} -- The quantity of shares to be purchased.

        asset_type {str} -- The instrument asset type. For example, `EQUITY`.

        Keyword Arguments:
        ----
        sub_asset_type {str} -- The instrument sub-asset type, not always needed. For example, `ETF`. (default: {None})

        Returns:
        ----
        {dict} -- A dictionary with the instrument.
        """

        leg = self.order["orderLegCollection"][order_leg_id]

        leg["instrument"]["symbol"] = symbol
        leg["instrument"]["assetType"] = asset_type
        leg["quantity"] = quantity

        self.order_size = quantity
        self.symbol = symbol
        self.asset_type = asset_type

        return leg

    def add_option_instrument(
        self, symbol: str, quantity: int, order_leg_id: int = 0
    ) -> dict:
        """Adds an Option instrument to the Trade object.

        Args:
        ----
        symbol (str): The option symbol to be added.

        quantity (int): The number of option contracts to purchase or sell.

        order_leg_id (int, optional): The position of the instrument within the
            the Order Leg Collection.. Defaults to 0.

        Returns:
        ----
        dict: The order leg containing the option contract.
        """

        self.instrument(
            symbol=symbol,
            quantity=quantity,
            asset_type="OPTION",
            order_leg_id=order_leg_id,
        )

        leg = self.order["orderLegCollection"][order_leg_id]

        return leg

    def good_till_cancel(self, cancel_time: datetime) -> None:
        """Converts an order to a `Good Till Cancel` order.

        Arguments:
        ----
        cancel_time {datetime.datetime} -- A datetime object representing the
            cancel time of the order.
        """

        self.order["duration"] = "GOOD_TILL_CANCEL"
        self.order["cancelTime"] = cancel_time.isoformat()

    def modify_side(self, side: str, leg_id: int = 0) -> None:
        """Modifies the Side the order takes.

        Arguments:
        ----
        side {str} -- The side to be set. Can be one of the following:
            `['buy', 'sell', 'sell_short', 'buy_to_cover']`.

        Keyword Arguments:
        ----
        leg_id {int} -- The leg you want to adjust. (default: {0})

        Raises:
        ----
        ValueError -- If the `side` argument does not match one of the valid sides,
            then a ValueError will be raised.
        """

        # Validate the Side.
        if side and side not in [
            "buy",
            "sell",
            "sell_short",
            "buy_to_cover",
            "sell_to_close",
            "buy_to_open",
        ]:
            raise ValueError(
                "The side you have specified is not valid. Please choose a valid side: ['buy', 'sell', 'sell_short', 'buy_to_cover','sell_to_close', 'buy_to_open']"
            )

        # Set the Order.
        if side:
            self.order["orderLegCollection"][leg_id]["instruction"] = side.upper()
        else:
            self.order["orderLegCollection"][leg_id]["instruction"] = (
                self.order_instructions[self.enter_or_exit][self.side_opposite]
            )

    def add_box_range(
        self,
        profit_size: float = 0.00,
        stop_size: float = 0.00,
        stop_percentage: bool = False,
        profit_percentage: bool = False,
        stop_limit: bool = False,
        make_one_cancels_other: bool = True,
        limit_size: float = 0.00,
        limit_percentage: bool = False,
    ):
        """Adds a Stop Loss(or Stop-Limit order), and a limit Order

        Arguments:
        ----
        profit_size {float} -- The size of desired profit. For example, `0.10`.

        profit_percentage {float} -- Specifies whether the `profit_size` is in absolute dollars `False` or
            in percentage terms `True`.

        stop_size {float} -- The size of desired stop loss. For example, `0.10`.

        stop_percentage {float} -- Specifies whether the `stop_size` is in absolute dollars `False` or
            in percentage terms `True`.

        Keyword Arguments:
        ----
        stop_limit {bool} -- If `True` makes the stop-loss a stop-limit. (default: {False})
        """

        if not self._triggered_added:
            self._convert_to_trigger()

        # Add a take profit Limit order.
        self.add_take_profit(profit_size=profit_size, percentage=profit_percentage)

        # Add a stop Loss Order.
        if not stop_limit:
            self.add_stop_loss(stop_size=profit_size, percentage=stop_percentage)
        else:
            self.add_stop_limit(
                stop_size=profit_size,
                limit_size=limit_size,
                stop_percentage=stop_percentage,
                limit_percentage=limit_percentage,
            )

        if make_one_cancels_other:
            self.add_one_cancels_other()

        self.is_box_range = True

    def add_stop_loss(self, stop_size: float, percentage: bool = False) -> bool:
        """Add's a stop loss order to exit the position when a certain loss is reached.

        Arguments:
        ----
        stop_size {float} -- The size of the stop from the current trading price. For example, `0.10`.

        Keyword Arguments:
        ----
        percentage {bool} -- Specifies whether the `stop_size` adjustment is a
            `percentage` or an `absolute dollar amount`. If `True` will calculate the
            stop size as a percentage of the current price. (default: {False})

        Returns:
        ----
        {bool} -- `True` if the order was added.
        """

        if not self._triggered_added:
            self._convert_to_trigger()

        price = self.grab_price()

        if percentage:
            adjustment = 1.0 - stop_size
            new_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=True
            )
        else:
            adjustment = -stop_size
            new_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=False
            )

        stop_loss_order = {
            "orderType": "STOP",
            "session": "NORMAL",
            "duration": "DAY",
            "stopPrice": new_price,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": self.order_instructions[self.enter_or_exit_opposite][
                        self.side
                    ],
                    "quantity": self.order_size,
                    "instrument": {"symbol": self.symbol, "assetType": self.asset_type},
                }
            ],
        }

        self.stop_loss_order = stop_loss_order
        self.order["childOrderStrategies"].append(self.stop_loss_order)

        return True

    def add_stop_limit(
        self,
        stop_size: float,
        limit_size: float,
        stop_percentage: bool = False,
        limit_percentage: bool = False,
    ):
        """Add's a Stop Limit Order to exit a trade when a stop price is reached but does not exceed the limit.

        Arguments:
        ----
        stop_size {float} -- The size of the stop from the current trading price. For example, `0.10`.

        limit_size {float} -- The size of the limit from the current stop price. For example, `0.10`.

        Keyword Arguments:
        ----
        stop_percentage {bool} -- Specifies whether the `stop_size` adjustment is a
            `percentage` or an `absolute dollar amount`. If `True` will calculate the
            stop size as a percentage of the current price. (default: {False})

        limit_percentage {bool} -- Specifies whether the `limit_size` adjustment is a
            `percentage` or an `absolute dollar amount`. If `True` will calculate the
            limit size as a percentage of the current stop price. (default: {False})

        Returns:
        ----
        {bool} -- `True` if the order was added.
        """

        # Check to see if there is a trigger.
        if not self._triggered_added:
            self._convert_to_trigger()

        price = self.grab_price()

        # Calculate the Stop Price.
        if stop_percentage:
            adjustment = 1.0 - stop_size
            stop_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=True
            )
        else:
            adjustment = -stop_size
            stop_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=False
            )

        # Calculate the Limit Price.
        if limit_percentage:
            adjustment = 1.0 - limit_size
            limit_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=True
            )
        else:
            adjustment = -limit_size
            limit_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=False
            )

        # Add the order.
        stop_limit_order = {
            "orderType": "STOP_LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "price": limit_price,
            "stopPrice": stop_price,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": self.order_instructions[self.enter_or_exit_opposite][
                        self.side
                    ],
                    "quantity": self.order_size,
                    "instrument": {"symbol": self.symbol, "assetType": self.asset_type},
                }
            ],
        }

        self.stop_limit_order = stop_limit_order
        self.order["childOrderStrategies"].append(self.stop_limit_order)

        return True

    def _calculate_new_price(
        self, price: float, adjustment: float, percentage: bool
    ) -> float:
        """Calculates an adjusted price given an old price.

        Arguments:
        ----
        price {float} -- The original price.

        adjustment {float} -- The adjustment to be made to the new price.

        percentage {bool} -- Specifies whether the adjustment is a percentage adjustment `True` or
            an absolute dollar adjustment `False`.

        Returns:
        ----
        {float} -- The new price after the adjustment has been made.
        """

        if percentage:
            new_price = price * adjustment
        else:
            new_price = price + adjustment

        # For orders below $1.00, can only have 4 decimal places.
        if new_price < 1:
            new_price = round(new_price, 4)

        # For orders above $1.00, can only have 2 decimal places.
        else:
            new_price = round(new_price, 2)

        return new_price

    def grab_price(self) -> float:
        """Grabs the current price of the order.

        Returns
        -------
        float
            The price rounded to 2 decimal places.
        """

        # We need to basis to calculate off of. Use the price.
        if self.order_type == "mkt":

            quote = self.api.get_quotes(instruments=[self.symbol])

            # Have to make a call to Get Quotes.
            price = quote[self.symbol]["lastPrice"]

        elif self.order_type == "lmt":
            price = self.price

        else:

            quote = self.api.get_quotes(instruments=[self.symbol])

            # Have to make a call to Get Quotes.
            price = quote[self.symbol]["lastPrice"]

        return round(price, 2)

    def add_take_profit(self, profit_size: float, percentage: bool = False) -> bool:
        """Add's a Limit Order to exit a trade when a profit threshold is reached.

        Arguments:
        ----
        profit_size {float} -- The size of the profit you want to make. For example, `0.10`.

        Keyword Arguments:
        ----
        percentage {bool} -- Specifies whether the `profit_size` passed through is a
            `percentage` or an `absolute dollar amount`. If `True` will calculate the
            profit as a percentage of the current price. (default: {False})

        Returns:
        ----
        {bool} -- `True` if the order was added.
        """

        # Check to see if we have a trigger order.
        if not self._triggered_added:
            self._convert_to_trigger()

        price = self.grab_price()

        # Calculate the new price.
        if percentage:
            adjustment = 1.0 + profit_size
            new_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=True
            )
        else:
            adjustment = profit_size
            new_price = self._calculate_new_price(
                price=price, adjustment=adjustment, percentage=False
            )

        # Build the order.
        take_profit_order = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "price": new_price,
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": self.order_instructions[self.enter_or_exit_opposite][
                        self.side
                    ],
                    "quantity": self.order_size,
                    "instrument": {"symbol": self.symbol, "assetType": self.asset_type},
                }
            ],
        }

        # Add the order.
        self.take_profit_order = take_profit_order
        self.order["childOrderStrategies"].append(self.take_profit_order)

        return True

    def add_one_cancels_other(self, orders: List[Dict] = None) -> Dict:
        """Add's a One Cancel's Other Order

        Arguments:
        ----
        orders {List[Dict]} -- A list of two orders that will cancel each other
            if one of those orders are executed.

        Returns:
        ----
        {Dict} -- A template that can be added to a Child Order Strategies list.
        """

        # Define the OCO Template
        new_temp = [{"orderStrategyType": "OCO", "childOrderStrategies": []}]

        # If we alread have a trigger than their are orders there.
        if self._triggered_added:

            # Grab the old ones.
            old_orders = self.order["childOrderStrategies"]

            # Add them to the template.
            new_temp[0]["childOrderStrategies"] = old_orders

            # Set the new child order strategy.
            self.order["childOrderStrategies"] = new_temp

        # Set it so we know it's a One Cancels Other.
        self._one_cancels_other = True

    def _convert_to_trigger(self):
        """Converts a regular order to a trigger order.

        Overview:
        ----
        Trigger orders can be used to have a stop loss orders, or take profit
        orders placed right after the main order has been placed. This helps
        protect the order when possible and take profit when thresholds are
        reached.
        """

        # Only convert to a trigger order, if it already isn't one.
        if self.order and not self._triggered_added:
            self.order["orderStrategyType"] = "TRIGGER"

            # Trigger orders will have child strategies, so initalize that list.
            self.order["childOrderStrategies"] = []

            # Update the state.
            self._triggered_added = True

    def modify_session(self, session: str) -> None:
        """Changes which session the order is for.

        Description
        ----
        Orders are able to be active during different trading sessions.
        If you would like the order to be active during a different session,
        then choose one of the following:

        1. 'am' - This is for pre-market hours.
        2. 'pm' - This is for post-market hours.
        3. 'normal' - This is for normal market hours.
        4. 'seamless' - This makes the order active all of the sessions.

        Arguments:
        ----
        session {str} -- The session you want the order to be active. Possible values
            are ['am', 'pm', 'normal', 'seamless']
        """

        if session in ["am", "pm", "normal", "seamless"]:
            self.order["session"] = session.upper()
        else:
            raise ValueError(
                "Invalid session, choose either am, pm, normal, or seamless"
            )

    @property
    def order_response(self) -> dict:
        """Returns the order response from submitting an order.

        Returns:
        ----
        {dict} -- The order response dictionary.
        """

        return self._order_response

    @order_response.setter
    def order_response(self, order_response_dict: dict) -> None:
        """Sets the order response from submitting an order.

        Arguments:
        ----
        order_response_dict {dict} -- The order response dictionary.
        """

        self._order_response = order_response_dict

    def _generate_order_id(self) -> str:
        """Generates an ID that can be used to identify the order.

        Returns:
        ----
        {str} -- The order ID that was generated.
        """

        # If we have an order, then generate it.
        if self.order:

            order_id = "{symbol}_{side}_{enter_or_exit}_{timestamp}"

            order_id = order_id.format(
                symbol=self.symbol,
                side=self.side,
                enter_or_exit=self.enter_or_exit,
                timestamp=datetime.now().timestamp(),
            )

            return order_id

        else:
            return ""

    def add_leg(
        self,
        order_leg_id: int,
        symbol: str,
        quantity: int,
        asset_type: str,
        sub_asset_type: str = None,
    ) -> List[dict]:
        """Adds an instrument to a trade.

        Arguments:
        ----
        order_leg_id {int} -- The position you want the new leg to be in the leg collection.

        symbol {str} -- The instrument ticker symbol.

        quantity {int} -- The quantity of shares to be purchased.

        asset_type {str} -- The instrument asset type. For example, `EQUITY`.

        Keyword Arguments:
        ----
        sub_asset_type {str} -- The instrument sub-asset type, not always needed. For example, `ETF`. (default: {None})

        Returns:
        ----
        {dict} -- The order's order leg collection.
        """

        # Define the leg.
        leg = {}
        leg["instrument"]["symbol"] = symbol
        leg["instrument"]["assetType"] = asset_type
        leg["quantity"] = quantity

        if sub_asset_type:
            leg["instrument"]["subAssetType"] = sub_asset_type

        # If 0, call instrument.
        if order_leg_id == 0:
            self.instrument(
                symbol=symbol,
                asset_type=asset_type,
                quantity=quantity,
                sub_asset_type=sub_asset_type,
                order_leg_id=0,
            )
        else:
            # Insert it.
            order_leg_colleciton: list = self.order["orderLegCollection"]
            order_leg_colleciton.insert(order_leg_id, leg)

        return self.order["orderLegCollection"]

    @property
    def number_of_legs(self) -> int:
        """Returns the number of legs in the Order Leg Collection.

        Returns:
        ----
        int: The count of legs in the collection.
        """

        return len(self.order["orderLegCollection"])

    def modify_price(self, new_price: float, price_type: str) -> None:
        """Used to change the price that is specified.

        Arguments:
        ----
        new_price (float): The new price to be set.

        price_type (str): The type of price that should be modified. Can
            be one of the following: [
                'price',
                'stop-price',
                'limit-price',
                'stop-limit-stop-price',
                'stop-limit-limit-price'
            ]
        """

        if price_type == "price":
            self.order["price"] = new_price
        elif price_type == "stop-price" and self.is_stop_order:
            self.order["stopPrice"] = new_price
            self.stop_price = new_price
        elif price_type == "limit-price" and self.is_limit_order:
            self.order["price"] = new_price
            self.price = new_price
        elif price_type == "stop-limit-limit-price" and self.is_stop_limit_order:
            self.order["price"] = new_price
            self.stop_limit_price = new_price
        elif price_type == "stop-limit-stop-price" and self.is_stop_limit_order:
            self.order["stopPrice"] = new_price
            self.stop_price = new_price

    @property
    def is_stop_order(self) -> bool:
        """Specifies whether the order is a Stop Loss Order.

        Returns:
        ----
        bool: `True` if the order is a Stop order, `False` otherwise.
        """

        if self.order_type != "stop":
            return False
        else:
            return True

    @property
    def is_stop_limit_order(self) -> bool:
        """Specifies whether the order is a Stop Limit Order.

        Returns:
        ----
        bool: `True` if the order is a Stop Limit order, `False` otherwise.
        """

        if self.order_type != "stop-lmt":
            return False
        else:
            return True

    @property
    def is_limit_order(self) -> bool:
        """Specifies whether the order is a Limit Order.

        Returns:
        ----
        bool: `True` if the order is a Limit order, `False` otherwise.
        """

        if self.order_type != "lmt":
            return False
        else:
            return True

    @property
    def is_trigger_order(self) -> bool:
        """Specifies whether the order is a trigger order.

        Returns:
        ----
        bool: `True` if the order is a contains a trigger,
            `False` otherwise.
        """

        if self._triggered_added:
            return True
        else:
            return False

    def _process_order_response(self) -> None:
        """Processes an order response, after is has been submitted."""

        self.order_id = self._order_response["order_id"]
        self.order_status = "QUEUED"

    def _update_order_status(self) -> None:
        """Updates the current order status, to reflect what's on TD."""

        if self.order_id != "":

            order_response = self.api.get_orders(
                account=self.account, order_id=self.order_id
            )

            self.order_response = order_response
            self.order_status = self.order_response["status"]

    def check_status(self) -> object:
        """Used to easily identify the order status.

        Returns
        -------
        OrderStatus
            An order status object that provides simple
            properties to grab the order status.
        """

        from pyrobot.order_status import OrderStatus

        return OrderStatus(trade_obj=self)

    def update_children(self) -> None:
        """Updates the Price info of the children info."""

        # Grab the children.
        children = self.order["childOrderStrategies"][0]["childOrderStrategies"]

        # Loop through each child.
        for order in children:

            # Get the latest price.
            quote = self.api.get_quotes(instruments=[self.symbol])
            last_price = quote[self.symbol]["lastPrice"]

            # Update the price.
            if order["orderType"] == "STOP":
                order["stopPrice"] = round(order["stopPrice"] + last_price, 2)
            elif order["orderType"] == "LIMIT":
                order["price"] = round(order["price"] + last_price, 2)

    async def run(self):
        results = {}
        for symbol in self.config.SYMBOLS:
            historical_data = await self.fetch_historical_data(symbol)
            results[symbol] = self.backtest_symbol(symbol, historical_data)
        return results

    async def fetch_historical_data(self, symbol):
        # Convert date strings to timestamps
        start_timestamp = int(
            datetime.strptime(self.config.BACKTEST_START_DATE, "%Y-%m-%d").timestamp()
        )
        end_timestamp = int(
            datetime.strptime(self.config.BACKTEST_END_DATE, "%Y-%m-%d").timestamp()
        )

        candles = await self.api.ticks_history(
            {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 5000,
                "end": end_timestamp,
                "start": start_timestamp,
                "style": "candles",
            }
        )
        return pd.DataFrame(candles["candles"])

    def backtest_symbol(self, symbol, data):
        """Backtest a single symbol using historical data."""
        balance = 10000
        trades = []

        for i in range(len(data)):
            if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                entry_price = data.iloc[i]["close"]
                stop_loss = entry_price - (self.config.STOP_LOSS_PIPS / 10000)
                take_profit = entry_price + (self.config.TAKE_PROFIT_PIPS / 10000)

                for j in range(i + 1, len(data)):
                    current_price = data.iloc[j]["close"]
                    if current_price <= stop_loss:
                        profit = (stop_loss - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break
                    elif current_price >= take_profit:
                        profit = (take_profit - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break

        return {
            "final_balance": balance,
            "trades": trades,
            "total_trades": len(trades),
            "profit": sum(trades),
        }
