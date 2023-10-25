import json

from enum import Enum


class PayloadAction(Enum):
    """Enumeration of the different possible actions embedded into a :class:`Payload`."""

    USER_MESSAGE = 'user_message'
    """PayloadAction: Indicates that the payload's purpose is to send a user message."""

    USER_VOICE = 'user_voice'
    """PayloadAction: Indicates that the payload's purpose is to send a user audio."""
    
    USER_FILE = 'user_file'
    """PayloadAction: Indicates that the payload's purpose is to send a user file."""

    RESET = 'reset'
    """PayloadAction: Use the :class:`~besser.bot.platforms.websocket.websocket_platform.WebSocketPlatform` on this
    bot.
    """

    BOT_REPLY_STR = 'bot_reply_str'
    """PayloadAction: Indicates that the payload's purpose is to send a bot reply containing a :class:`str` object."""

    BOT_REPLY_FILE = 'bot_reply_file'
    """PayloadAction: Indicates that the payload's purpose is to send a bot file containing a :class:`dict` object."""

    BOT_REPLY_DF = 'bot_reply_dataframe'
    """PayloadAction: Indicates that the payload's purpose is to send a bot reply containing a :class:`pandas.DataFrame`
    object.
    """

    BOT_REPLY_OPTIONS = 'bot_reply_options'
    """PayloadAction: Indicates that the payload's purpose is to send a bot reply containing a list of strings, where 
    the user should select 1 of them.
    """


class Payload:
    """Represents a payload object used for encoding and decoding messages between a bot and any other external agent.
    """

    @staticmethod
    def decode(payload_str):
        """Decode a JSON payload string into a :class:`Payload` object.

        Args:
            payload_str (str): A JSON-encoded payload string.

        Returns:
            Payload or None: A Payload object if the decoding is successful,
            None otherwise.
        """
        payload_dict = json.loads(payload_str)
        payload_action = payload_dict['action']
        payload_message = payload_dict['message']
        for action in PayloadAction:
            if action.value == payload_action:
                return Payload(action, payload_message)
        return None

    def __init__(self, action: PayloadAction, message: str = None):
        self.action: str = action.value
        self.message: str = message


class PayloadEncoder(json.JSONEncoder):
    """Encoder for the :class:`Payload` class.

    Example:
        .. code::

            import json
            encoded_payload = json.dumps(payload, cls=PayloadEncoder)
    """

    def default(self, obj):
        """Returns a serializable object for a :class:`Payload`

        Args:
            obj: the object to serialize

        Returns:
            dict: the serialized payload
        """
        if isinstance(obj, Payload):
            # Convert the Payload object to a dictionary
            payload_dict = {
                'action': obj.action,
                'message': obj.message,
            }
            return payload_dict
        return super().default(obj)
