# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for evaluation

import argparse
import time
from prettytable import PrettyTable
from rgbt import LasHeR, RGBT234, RGBT210, GTOT


class DefaultEvaluation:
    def __init__(self, tracker_names: list, result_paths: list, plot_curve=False, plot_radar=False):
        assert len(tracker_names) == len(result_paths)
        self.tracker_names = tracker_names
        self.result_paths = result_paths
        self.curve = plot_curve
        self.radar = plot_radar

    def print_metrics(self):
        ...

    def plot_curve(self):
        ...

    def plot_radar(self):
        ...

    def run(self):
        start_time = time.time()
        print('Evaluating...')
        self.print_metrics()
        if self.curve:
            print('Plotting curve...')
            self.plot_curve()
        if self.radar:
            print('Plotting attribute radar...')
            self.plot_radar()
        print(f'Evalutation time: {time.time() - start_time: .2f}s')


class LasHeREvaluation(DefaultEvaluation):
    def __init__(self, tracker_names: list, result_paths: list, plot_curve=False, plot_radar=False):
        super(LasHeREvaluation, self).__init__(tracker_names, result_paths, plot_curve, plot_radar)
        self.evaluator = LasHeR()

        for tracker_name, result_path in zip(tracker_names, result_paths):
            self.evaluator(tracker_name, result_path)

    def print_metrics(self):
        npr_dict = self.evaluator.NPR()
        pr_dict = self.evaluator.PR()
        sr_dict = self.evaluator.SR()

        table = PrettyTable(['Tracker', 'NPR(%)', 'PR(%)', 'SR(%)'])
        for tracker_name in self.tracker_names:
            table.add_row([tracker_name, round(npr_dict[tracker_name][0] * 100, 1),
                           round(pr_dict[tracker_name][0] * 100, 1),
                           round(sr_dict[tracker_name][0] * 100, 1)])

        print(table)

    def plot_curve(self):
        self.evaluator.draw_plot(metric_fun=self.evaluator.NPR)
        self.evaluator.draw_plot(metric_fun=self.evaluator.PR)
        self.evaluator.draw_plot(metric_fun=self.evaluator.SR)

    def plot_radar(self):
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.NPR)
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.PR)
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.SR)


class RGBT234Evaluation(DefaultEvaluation):
    def __init__(self, tracker_names: list, result_paths: list, plot_curve=False, plot_radar=False):
        super(RGBT234Evaluation, self).__init__(tracker_names, result_paths, plot_curve, plot_radar)
        self.evaluator = RGBT234()

        for tracker_name, result_path in zip(tracker_names, result_paths):
            self.evaluator(tracker_name, result_path)

    def print_metrics(self):
        pr_dict = self.evaluator.MPR()
        sr_dict = self.evaluator.MSR()

        table = PrettyTable(['Tracker', 'MPR(%)', 'MSR(%)'])
        for tracker_name in self.tracker_names:
            table.add_row([tracker_name,
                           round(pr_dict[tracker_name][0] * 100, 1),
                           round(sr_dict[tracker_name][0] * 100, 1)])

        print(table)

    def plot_curve(self):
        self.evaluator.draw_plot(metric_fun=self.evaluator.MPR)
        self.evaluator.draw_plot(metric_fun=self.evaluator.MSR)

    def plot_radar(self):
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.MPR)
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.MSR)


class RGBT210Evaluation(DefaultEvaluation):
    def __init__(self, tracker_names: list, result_paths: list, plot_curve=False, plot_radar=False):
        super(RGBT210Evaluation, self).__init__(tracker_names, result_paths, plot_curve, plot_radar)
        self.evaluator = RGBT210()

        for tracker_name, result_path in zip(tracker_names, result_paths):
            self.evaluator(tracker_name, result_path)

    def print_metrics(self):
        pr_dict = self.evaluator.PR()
        sr_dict = self.evaluator.SR()

        table = PrettyTable(['Tracker', 'PR(%)', 'SR(%)'])
        for tracker_name in self.tracker_names:
            table.add_row([tracker_name,
                           round(pr_dict[tracker_name][0] * 100, 1),
                           round(sr_dict[tracker_name][0] * 100, 1)])

        print(table)

    def plot_curve(self):
        self.evaluator.draw_plot(metric_fun=self.evaluator.PR)
        self.evaluator.draw_plot(metric_fun=self.evaluator.SR)

    def plot_radar(self):
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.PR)
        self.evaluator.draw_attributeRadar(metric_fun=self.evaluator.SR)


class GTOTEvaluation(DefaultEvaluation):
    def __init__(self, tracker_names: list, result_paths: list, plot_curve=False, plot_radar=False):
        super(GTOTEvaluation, self).__init__(tracker_names, result_paths, plot_curve, plot_radar)
        self.evaluator = GTOT()

        for tracker_name, result_path in zip(tracker_names, result_paths):
            self.evaluator(tracker_name, result_path)

    def print_metrics(self):
        pr_dict = self.evaluator.MPR()
        sr_dict = self.evaluator.MSR()

        table = PrettyTable(['Tracker', 'MPR(%)', 'MSR(%)'])
        for tracker_name in self.tracker_names:
            table.add_row([tracker_name,
                           round(pr_dict[tracker_name][0] * 100, 1),
                           round(sr_dict[tracker_name][0] * 100, 1)])

        print(table)

    def plot_curve(self):
        self.evaluator.draw_plot(metric_fun=self.evaluator.MPR)
        self.evaluator.draw_plot(metric_fun=self.evaluator.MSR)

    def plot_radar(self):
        raise NotImplementedError('Attribute radar on GTOT is not implemented yet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['lasher', 'rgbt234', 'rgbt210', 'gtot'])
    parser.add_argument('--tracker_names', type=str, help='Tracker names (separated by comma)', nargs='+')
    parser.add_argument('--result_paths', type=str, help='Result paths (separated by comma)', nargs='+')
    parser.add_argument('--plot_curve', action='store_true', help='Plot metrics')
    parser.add_argument('--plot_radar', action='store_true', help='Plot metrics')

    args = parser.parse_args()

    if args.dataset == 'lasher':
        LasHeREvaluation(args.tracker_names, args.result_paths, args.plot_curve, args.plot_radar).run()
    elif args.dataset == 'rgbt234':
        RGBT234Evaluation(args.tracker_names, args.result_paths, args.plot_curve, args.plot_radar).run()
    elif args.dataset == 'rgbt210':
        RGBT210Evaluation(args.tracker_names, args.result_paths, args.plot_curve, args.plot_radar).run()
    elif args.dataset == 'gtot':
        GTOTEvaluation(args.tracker_names, args.result_paths, args.plot_curve, args.plot_radar).run()
