# Network

Send and receive events over UDP. The wire format is compatible with
[aestream](https://github.com/aestream/aestream) and SpiNNaker's SPIF.

```python
import eventcv as ecv

sender = ecv.UdpSender("127.0.0.1:3333")
with ecv.stream(dt_ms=10) as cam:
    for events in cam:
        sender.send(events)
```

```python
receiver = ecv.UdpReceiver("0.0.0.0:3333", sensor_size=(640, 480))
while True:
    events = receiver.recv(dt_ms=30)  # an empty stream means nothing arrived
    frame = events.tencode()
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `timestamps` | `False` | Send a timestamp with every event. Doubles bandwidth. Without it, the receiver stamps events on arrival. |
| `host_endian` | `True` | Use host byte order, as aestream does. Set `False` for network byte order. |

The receiver detects whether timestamps are present, so it doesn't need to match the sender's
`timestamps` setting.

:::{warning}
UDP doesn't confirm delivery. `send()` returns the number of events handed to the socket, not the
number received. Run the receiver concurrently with the sender, or a burst will overflow its buffer.
:::

## Wire format

Each event is a 32-bit word: `x` in the high 16 bits, `y` in the low 16, polarity in bit 15, and bit
31 set when a 32-bit timestamp word follows. Timestamps are microseconds and wrap after about 71
minutes.
