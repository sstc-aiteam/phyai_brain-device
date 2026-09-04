ROBOTIQ_PREAMBLE = r"""
# ============================================================
# Robotiq 2F - Minimal URScript Helper
# ============================================================

def rq_get_var(name):
    local socket_name = "rq_socket"

    local connected = socket_open(
        "127.0.0.1",
        63352,
        socket_name
    )

    if not connected:
        return -1
    end

    local value = socket_get_var(
        name,
        socket_name
    )

    socket_close(
        socket_name
    )

    return value
end


def rq_set_var(name, value):
    local socket_name = "rq_socket"

    local connected = socket_open(
        "127.0.0.1",
        63352,
        socket_name
    )

    if not connected:
        return False
    end

    local result = socket_set_var(
        name,
        value,
        socket_name
    )

    # Robotiq's daemon may return ACK. Because this helper closes
    # the socket after each SET, any ACK cannot interfere with a
    # later GET operation.
    sleep(0.01)

    socket_close(
        socket_name
    )

    return result
end


# ============================================================
# Activation
# ============================================================

def rq_reset():
    rq_set_var(
        "GTO",
        0
    )

    rq_set_var(
        "ACT",
        0
    )

    sleep(0.10)

    return True
end


def rq_activate():
    rq_set_var(
        "ACT",
        1
    )

    rq_set_var(
        "GTO",
        1
    )

    return True
end


def rq_activate_and_wait():
    rq_activate()

    local sta = rq_get_var(
        "STA"
    )

    local count = 0

    while (
        sta != 3
        and count < 250
    ):
        sleep(0.02)

        sta = rq_get_var(
            "STA"
        )

        count = count + 1
    end

    return (
        sta == 3
    )
end


# ============================================================
# Configuration
# ============================================================

def rq_set_speed(speed):
    if speed < 0:
        speed = 0
    end

    if speed > 255:
        speed = 255
    end

    return rq_set_var(
        "SPE",
        floor(speed)
    )
end


def rq_set_force(force):
    if force < 0:
        force = 0
    end

    if force > 255:
        force = 255
    end

    return rq_set_var(
        "FOR",
        floor(force)
    )
end


# ============================================================
# Position / Motion
# ============================================================

def rq_current_pos():
    return rq_get_var(
        "POS"
    )
end


def rq_move(position):
    if position < 0:
        position = 0
    end

    if position > 255:
        position = 255
    end

    rq_set_var(
        "POS",
        floor(position)
    )

    rq_set_var(
        "GTO",
        1
    )

    return True
end


def rq_move_and_wait(position):
    rq_move(
        position
    )

    local obj = rq_get_var(
        "OBJ"
    )

    local count = 0

    # OBJ:
    # 0 = fingers are moving
    # 1 = object detected while opening
    # 2 = object detected while closing
    # 3 = requested position reached
    while (
        obj == 0
        and count < 500
    ):
        sleep(0.02)

        obj = rq_get_var(
            "OBJ"
        )

        count = count + 1
    end

    return (
        obj == 1
        or obj == 2
        or obj == 3
    )
end


def rq_stop():
    rq_set_var(
        "GTO",
        0
    )

    return True
end


# ============================================================
# Status
# ============================================================

def rq_is_gripper_activated():
    local sta = rq_get_var(
        "STA"
    )

    return (
        sta == 3
    )
end


def rq_is_object_detected():
    local obj = rq_get_var(
        "OBJ"
    )

    return (
        obj == 1
        or obj == 2
    )
end


def rq_is_motion_complete():
    local obj = rq_get_var(
        "OBJ"
    )

    return (
        obj == 1
        or obj == 2
        or obj == 3
    )
end
"""

