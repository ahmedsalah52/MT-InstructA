from simple_pid import PID


def update(pos,desired):
    pid = [PID(1, 0.1, 0.05, setpoint=v) for v in desired]

    control = [pid[i](pos[i]) for i in range(len(desired))]

    return np.array(control)


