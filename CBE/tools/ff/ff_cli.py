"""FF command-line entry point. Live calibration starts only with `calib`."""

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parser():
    result = argparse.ArgumentParser(prog='ff', description='Fulcrum calibration tools')
    commands = result.add_subparsers(dest='command', required=True)
    calib = commands.add_parser('calib', help='calibrate all devices, or one camera')
    calib.add_argument('-c_i', '-c', '--camera', type=int, choices=(0, 1, 2))
    calib.add_argument('--no-kinect', action='store_true',
                       help='legacy webcam + projector rig calibration only')
    calib.add_argument('--restore', action='store_true',
                       help='restore a stored TD session after forced termination')
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    for i, argument in enumerate(arguments):
        if argument in ('-c_0', '-c_1', '-c_2'):
            arguments[i:i + 1] = ['-c_i', argument[-1]]
            break
    args = parser().parse_args(arguments)
    if args.camera is not None and (args.no_kinect or args.restore):
        parser().error('--camera cannot be combined with --no-kinect or --restore')
    try:
        import ff_calibrate as calibration
        from ff_calibration_io import (
            OutputSession, PIPE, SESSION_KEY, camera_record, evaluate_json,
            publish_documents, restore_output,
        )
        if args.restore:
            snapshot = evaluate_json(
                calibration, "op('/project1').storage.get(%r)" % SESSION_KEY)
            if snapshot is None:
                raise RuntimeError('no FF calibration session needs restoration')
            restore_output(calibration, snapshot)
            print('TouchDesigner selection and parameter bindings restored')
            return 0
        with OutputSession(calibration):
            if args.camera is None:
                calibration.run_rig(kinect=not args.no_kinect)
            else:
                record = calibration.calibrate_camera(args.camera)
                document = record if args.camera == 2 else camera_record(
                    calibration, args.camera, record)
                publish_documents({'camera_c%d.json' % args.camera: document})
                print('camera c%d -> %s' % (
                    args.camera, PIPE / ('camera_c%d.json' % args.camera)))
                print('Joint rig, projector and other camera records are unchanged')
        return 0
    except KeyboardInterrupt:
        print('ff: interrupted; output restored', file=sys.stderr)
        return 130
    except Exception as error:
        print('ff: error: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
