import click, os
import json
from datetime import datetime

@click.command()
@click.option('--data', '-d',
			  prompt='Data path',
			  help='Path of \'.npy\' file or name defined on --config file.')
@click.option('--outdir', '-o',
			  prompt='Output directory',
			  help='Directory to save/load model files.')
@click.option('--bs', default=64, help='Batch size (default=64).')
@click.option('--epochs', type=int)
@click.option('--test-each', default=5, help='Logs loss on full test and train set each x epochs (default=5, disable=-1).')
@click.option('--opt', default='voxels-usegan', help='Model arquitecture.')
@click.option('--extend-train/--no-extend-train', default=True,
              help='The model will train for additional epochs to ensure optimal model output.')
@click.option('--max-extend', default=10, help='Maximum number of iterations to do in the extended train.')
@click.option('--overwrite', is_flag=True)
@click.option('--workers', '-w', default=-1,
			  help='Number of processors to use while loading data (default=-1).')
@click.option('--training-ratio', '-tr', default=5, help='(default=5)')
@click.option('--gradient-penalty', '-gp', default=10, help='(default=10)')
@click.option('--loss', default='l1', help='l1 or l2 (default=l1)')
@click.option('--loss-multiply', default=100, help='(default=100)')
@click.option('--config', default='config.txt',
			  help='Directory with shortnames for data paths (default=\'config.txt\')\nex: modelnet10: /path/to/file.npy')
@click.option('--force', '-f', is_flag=True)
def main(data, outdir, bs, workers, training_ratio, gradient_penalty, loss, loss_multiply,
         overwrite, config, epochs, test_each, opt, extend_train, max_extend, force, 
         frac_ratio=None):

    if os.path.isdir(data):
        specs = json.load(open(os.path.join(data, "specs.json")))
        print("Loaded from specs file")
        if "opt" in specs:
            opt = specs["opt"]
            print("Using opt: {} from specs".format(opt))
        if "frac_ratio" in specs:
            frac_ratio = specs["frac_ratio"]
            print("using frac_ratio: {} from specs".format(frac_ratio))
        data = specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"])
        print("Loading data from {}".format(data))
    
    if overwrite and (not force):
        if os.path.isdir(outdir) and (len(os.listdir(outdir)) != 0):
            print("Outdir has the following files in it: {}".format(os.listdir(outdir)))
            click.confirm('Overwrite=True, Do you want to continue?', abort=True)
        
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    from .model import LoadModel
    
    out_log = os.path.join(outdir, 'exec.log')
    get_log = lambda x: datetime.now().strftime('%Y-%m-%d %H:%M:%S - ') + x
    log = get_log('BEGIN TRAINING\n')
    print('-'*40)
    
    # try:
    model = LoadModel(data, outdir, bs,
                        num_workers=workers,
                        training_ratio=training_ratio,
                        gradient_penalty=gradient_penalty,
                        loss=loss,
                        loss_multiply=loss_multiply,
                        overwrite=overwrite,
                        config_file=config,
                        shuffle_data=True,
                        opt=opt, 
                        frac_ratio=frac_ratio)
    model.train(epochs=epochs, test_each=test_each,
                extend_training=extend_train, max_extend=max_extend)
    # except Exception as e:
    #     log += get_log(f'TRAINING ENDED WITH ERROR: {e!r}\n')
    #     traceback.print_exc()
    #     print(e)
    #     print('TRAINING ENDED WITH ERROR')
    # else:
    #     log += get_log('TRAINING ENDED\n')
    
    if os.path.exists(outdir):
        with open(out_log, 'a') as f:
            f.writelines(log)
        
    print('-'*40)
        
if __name__ == '__main__':
    main()
