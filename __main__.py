import sys
import os
import argparse

# Update PATH to include the local DIGITS directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
found_parent_dir = False
for p in sys.path:
    if os.path.abspath(p) == PARENT_DIR:
        found_parent_dir = True
        break
if not found_parent_dir:
    sys.path.insert(0, PARENT_DIR)


def main():
    parser = argparse.ArgumentParser(description='DIGITS server')
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8181,
        help='Port to run app on (default 8181)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help=('Run the application in debug mode (reloads when the source '
              'changes and gives more detailed error messages)')
    )

    args = vars(parser.parse_args())

    import easyclassify
    import easyclassify.config
    import easyclassify.log
    import easyclassify.app

    try:
        if not easyclassify.app.scheduler.start():
            print('ERROR: Scheduler would not start')
        else:
            easyclassify.app.socketio.run(easyclassify.app.app, '0.0.0.0', args['port'])
    except KeyboardInterrupt:
        pass
    finally:
        easyclassify.app.scheduler.stop()


if __name__ == '__main__':
    main()
