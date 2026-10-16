import socket

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_unit_tests_cannot_open_network_connections() -> None:
    with pytest.raises((SocketBlockedError, SocketConnectBlockedError)):
        socket.create_connection(("203.0.113.1", 80), timeout=1)
