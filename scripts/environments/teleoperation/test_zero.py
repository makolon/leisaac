# zmq_pose_subscriber.py
import argparse
import json
import time
import zmq


def main():
    parser = argparse.ArgumentParser(description="Simple ZeroMQ SUB tester for 'pose {json}\\n'")
    parser.add_argument("--host", default="0.0.0.0", help="Bind/Connect host (bind: interface, connect: remote host)")
    parser.add_argument("--port", type=int, default=5555, help="Port number")
    parser.add_argument("--topic", default="pose", help="Topic prefix to subscribe")
    parser.add_argument("--mode", choices=["bind", "connect"], default="bind",
                        help="bind: SUB.bind(tcp://host:port), connect: SUB.connect(tcp://host:port)")
    parser.add_argument("--timeout", type=float, default=0.5, help="Poll timeout in seconds")
    parser.add_argument("--max", type=int, default=0, help="Stop after N messages (0 = infinite)")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON instead of pretty summary")
    parser.add_argument("--subscribe_all", action="store_true", help="Subscribe to all topics (empty prefix)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose debug information")
    args = parser.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    
    # Set socket options for better reliability
    sock.setsockopt(zmq.RCVHWM, 1000)  # Set high water mark for incoming messages
    sock.setsockopt(zmq.LINGER, 0)  # Don't wait when closing

    # Topic subscription (prefix match). Empty string = all topics.
    topic = "" if args.subscribe_all else args.topic
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    if args.mode == "bind":
        # Your publisher uses PUB.connect(...), so SUB.bind(...) is the natural pair here.
        sock.bind(endpoint)
        print(f"[SUB] bind    -> {endpoint} (topic='{topic}')")
        if args.verbose:
            print(f"[SUB] Listening on all interfaces (0.0.0.0) port {args.port}")
            print(f"[SUB] Sender should connect to: tcp://<this-machine-ip>:{args.port}")
    else:
        sock.connect(endpoint)
        print(f"[SUB] connect -> {endpoint} (topic='{topic}')")

    # Give the subscription a moment to propagate so early messages aren't missed.
    # (In PUB/SUB, subscriptions are asynchronous control messages.)
    time.sleep(0.5)  # Increased from 0.2 to 0.5 for better reliability
    
    if args.verbose:
        print(f"[SUB] Socket configured:")
        print(f"      RCVHWM: {sock.getsockopt(zmq.RCVHWM)}")
        print(f"      LINGER: {sock.getsockopt(zmq.LINGER)}")
        print(f"      Subscriptions: '{topic}'")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    count = 0
    print("[SUB] waiting for messages ... Ctrl-C to stop")
    if args.verbose:
        print(f"[SUB] Poll timeout: {args.timeout}s")
    
    last_heartbeat = time.time()
    heartbeat_interval = 5.0  # Print heartbeat every 5 seconds
    
    try:
        while True:
            events = dict(poller.poll(int(args.timeout * 1000)))
            
            # Heartbeat for verbose mode
            if args.verbose and (time.time() - last_heartbeat) > heartbeat_interval:
                print(f"[SUB] Still waiting... (no messages for {heartbeat_interval}s)")
                last_heartbeat = time.time()
            
            if sock in events and events[sock] == zmq.POLLIN:
                line = sock.recv_string()  # we expect "pose {json}\n"
                line = line.strip()
                # Split "topic payload" once; fall back to raw JSON if no space.
                if " " in line:
                    recv_topic, payload = line.split(" ", 1)
                else:
                    recv_topic, payload = "", line

                # Try parse JSON
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON decode error: {e}: {payload[:120]}...")
                    continue

                now = time.time()
                ts = data.get("timestamp", None)
                latency_ms = (now - ts) * 1000.0 if isinstance(ts, (int, float)) else None

                if args.raw:
                    print(f"\n[{count+1}] topic='{recv_topic}' latency_ms={latency_ms:.2f}" if latency_ms is not None else f"\n[{count+1}] topic='{recv_topic}'")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    la = data.get("left_arm", {})
                    ra = data.get("right_arm", {})
                    lp = la.get("position", {})
                    rp = ra.get("position", {})
                    lg = la.get("gripper", None)
                    rg = ra.get("gripper", None)

                    print(f"\n[{count+1}] topic='{recv_topic}'", end="")
                    if latency_ms is not None:
                        print(f"  latency={latency_ms:.2f} ms")
                    else:
                        print()

                    print(f"  LEFT : pos=({lp.get('px'):.4f}, {lp.get('py'):.4f}, {lp.get('pz'):.4f})  grip={lg}")
                    print(f"  RIGHT: pos=({rp.get('px'):.4f}, {rp.get('py'):.4f}, {rp.get('pz'):.4f})  grip={rg}")

                count += 1
                if args.max and count >= args.max:
                    print("[SUB] reached max messages, exiting.")
                    break
            else:
                # Poll timeout tick; print a heartbeat if you like
                # print(".", end="", flush=True)
                pass
    except KeyboardInterrupt:
        print("\n[SUB] interrupted by user")
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
