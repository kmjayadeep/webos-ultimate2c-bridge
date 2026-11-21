"""Multi-receiver support (one 2.4 GHz receiver per controller).

Design:
- find_receivers() returns every matching receiver as (syspath, bus, dev), sorted by syspath.
- led_packet(player) returns the Xbox 360 set-LED packet for player 1-4 (01 03 02..05).
- Receiver(bus, dev, player=1) sends that player's LED packet in handshake().
- serve(syspath, bus, dev, player=1) passes the player on to Receiver.
- main() serves each receiver in its own thread at the same time. Each gets the lowest free
  player slot, which is freed when its serve() returns. A receiver leaving or failing does not
  disturb the others, and on stop main() waits for every serve() to return.
- At most MAX_PLAYERS (4) receivers are served; extra ones wait for a free slot.
"""
import errno, os, threading, time, unittest
from unittest import mock

from test_bridge import Base, FakeReceiver, FakeUinput, FindReceiver, ReceiverTest, ServeTest, bridge, report


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond(): return True
        time.sleep(0.005)
    return cond()


class FindReceivers(FindReceiver):
    def find_all(self):
        with mock.patch.object(bridge.glob, "glob", lambda pat: [
                os.path.join(self.tmp.name, n) for n in reversed(sorted(os.listdir(self.tmp.name))) if not n.endswith(".log")]):
            return bridge.find_receivers()

    def test_returns_all_matching_sorted(self):
        b = self.make_dev("1-2", busnum=1, devnum=8)
        self.make_dev("1-3", pid="3106")
        a = self.make_dev("1-1", busnum=1, devnum=7)
        c = self.make_dev("2-1", busnum=2, devnum=3)
        self.assertEqual(self.find_all(), [(a, 1, 7), (b, 1, 8), (c, 2, 3)])


class LedPacket(unittest.TestCase):
    def test_players(self):
        self.assertEqual([bridge.led_packet(p) for p in (1, 2, 3, 4)],
                         [b"\x01\x03\x02", b"\x01\x03\x03", b"\x01\x03\x04", b"\x01\x03\x05"])

    def test_player_one_matches_known_working_packet(self):
        self.assertEqual(bridge.led_packet(1), bridge.LED_PACKET)

    def test_out_of_range(self):
        for p in (0, 5):
            with self.subTest(player=p), self.assertRaises(ValueError): bridge.led_packet(p)


class ReceiverPlayer(ReceiverTest):
    def test_default_player_is_one(self):
        bridge.Receiver(1, 5).handshake()
        self.assertEqual(self.calls[-1][:3], ("bulk_out", bridge.EP_OUT, b"\x01\x03\x02"))

    def test_handshake_uses_player_led(self):
        bridge.Receiver(1, 5, player=3).handshake()
        self.assertEqual(self.calls[-1][:3], ("bulk_out", bridge.EP_OUT, b"\x01\x03\x04"))


class ServePlayer(ServeTest):
    def test_player_passed_to_receiver(self):
        rx = FakeReceiver(self, [report()]); ui = FakeUinput(); made = []
        def make(bus, dev, player=1): made.append((bus, dev, player)); return rx
        with mock.patch.object(bridge, "Receiver", make), mock.patch.object(bridge, "Uinput", lambda *a: ui):
            bridge.serve(self.sys, 1, 5, player=2)
        self.assertEqual(made, [(1, 5, 2)])


class FakeServe:
    """Stands in for serve(): blocks until released (receiver gone) or the bridge stops."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active, self.history, self.release, self.fail = {}, [], {}, set()
        self.linger = 0.0    # seconds serve() keeps running after the bridge is told to stop

    def __call__(self, syspath, bus, dev, player=1):
        key = (syspath, bus, dev); ev = threading.Event()
        with self.lock:
            self.active[key] = player; self.history.append((key, player)); self.release[key] = ev
        try:
            if key in self.fail: raise OSError(errno.EBUSY, "busy")
            while bridge.running and not ev.wait(0.005): pass
            if not bridge.running: time.sleep(self.linger)
        finally:
            with self.lock: del self.active[key]

    def players(self):
        with self.lock: return dict(self.active)


class MainMulti(Base):
    A, B, C = ("/sys/1-1", 1, 5), ("/sys/1-2", 1, 6), ("/sys/1-3", 1, 7)

    def setUp(self):
        super().setUp()
        self.present, self.serve = [], FakeServe()
        real_sleep = time.sleep
        for target, attr, val in ((bridge, "find_receivers", lambda: list(self.present)),
                                  (bridge, "serve", self.serve), (bridge.signal, "signal", lambda *a: None),
                                  (bridge.time, "sleep", lambda s: real_sleep(0.005))):
            p = mock.patch.object(target, attr, val); p.start(); self.addCleanup(p.stop)
        self.thread = threading.Thread(target=bridge.main, daemon=True)
        self.addCleanup(self.stop)

    def start(self): self.thread.start()

    def stop(self):
        bridge.running = False
        if self.thread.is_alive(): self.thread.join(2.0)

    def unplug(self, dev):
        self.present.remove(dev); self.serve.release[dev].set()

    def test_two_receivers_served_concurrently(self):
        self.present[:] = [self.A, self.B]; self.start()
        self.assertTrue(wait_until(lambda: len(self.serve.players()) == 2), self.serve.history)
        self.assertEqual(self.serve.players(), {self.A: 1, self.B: 2})

    def test_hotplugged_receiver_gets_next_player(self):
        self.present[:] = [self.A]; self.start()
        self.assertTrue(wait_until(lambda: self.serve.players() == {self.A: 1}))
        self.present.append(self.B)
        self.assertTrue(wait_until(lambda: self.serve.players() == {self.A: 1, self.B: 2}), self.serve.players())

    def test_unplug_leaves_others_running_and_frees_slot(self):
        self.present[:] = [self.A, self.B]; self.start()
        self.assertTrue(wait_until(lambda: len(self.serve.players()) == 2))
        self.unplug(self.A)
        self.assertTrue(wait_until(lambda: self.serve.players() == {self.B: 2}), self.serve.players())
        self.present.append(self.C)
        self.assertTrue(wait_until(lambda: self.serve.players() == {self.B: 2, self.C: 1}), self.serve.players())

    def test_failed_receiver_does_not_affect_others(self):
        self.serve.fail.add(self.A)
        self.present[:] = [self.A, self.B]; self.start()
        self.assertTrue(wait_until(lambda: self.B in self.serve.players()))
        self.assertTrue(wait_until(lambda: "serve failed" in self.logged()))
        self.assertIn(self.B, self.serve.players())

    def test_instance_not_reserved_while_still_present(self):
        self.present[:] = [self.A, self.B]; self.start()
        self.assertTrue(wait_until(lambda: len(self.serve.players()) == 2))
        self.serve.release[self.A].set()                     # serve returned (e.g. IDLE) but device still listed
        self.assertTrue(wait_until(lambda: self.A not in self.serve.players()))
        time.sleep(0.1)
        self.assertEqual([k for k, _ in self.serve.history].count(self.A), 1)
        self.present.remove(self.A); self.present.append(("/sys/1-1", 1, 9))   # re-enumerated: new devnum
        self.assertTrue(wait_until(lambda: ("/sys/1-1", 1, 9) in self.serve.players()))

    def test_fifth_receiver_waits_for_free_slot(self):
        devs = [("/sys/1-%d" % i, 1, 10 + i) for i in range(1, 6)]
        self.present[:] = devs; self.start()
        self.assertTrue(wait_until(lambda: len(self.serve.players()) == 4), self.serve.players())
        time.sleep(0.1)
        self.assertEqual(self.serve.players(), {d: i + 1 for i, d in enumerate(devs[:4])})
        self.assertEqual(self.logged().count("waiting: all 4 player slots in use"), 1)   # logged once, not every scan
        self.unplug(devs[1])
        self.assertTrue(wait_until(lambda: self.serve.players().get(devs[4]) == 2), self.serve.players())

    def test_stop_waits_for_all_serves(self):
        self.serve.linger = 0.2
        self.present[:] = [self.A, self.B]; self.start()
        self.assertTrue(wait_until(lambda: len(self.serve.players()) == 2))
        bridge.running = False; self.thread.join(2.0)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.serve.players(), {})
        self.assertTrue(self.logged().rstrip().endswith("bridge stopped"))


def load_tests(loader, tests, pattern):
    """Run only the tests defined here, not the ones inherited from test_bridge's classes."""
    suite = unittest.TestSuite()
    for cls in (FindReceivers, LedPacket, ReceiverPlayer, ServePlayer, MainMulti):
        suite.addTests(cls(n) for n in loader.getTestCaseNames(cls) if n in vars(cls))
    return suite


if __name__ == "__main__":
    unittest.main()
