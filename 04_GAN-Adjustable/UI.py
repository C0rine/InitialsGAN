from IPython.display import HTML, display, Markdown, Javascript
import ipywidgets as widgets

# initialize the values
lossval = 'BCE'
optimval = 'Adam'
normval = 'BatchNorm'
batchval = '32'
epochval = 25
lrateval = 0.0002
beta1val = 0.5
clampval = 0.01
savepathval = 'Z:/GAN-Training-Results/'

formattedsettings = ''

# initialize the widgets
loss = widgets.ToggleButtons(
    options=['BCE', 'Wasserstein'],
    value = lossval,
    description='Loss:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

optim = widgets.ToggleButtons(
    options=['Adam', 'RMSprop'],
    value=optimval,
    description='Optimizer:',
    disabled=False,
)

norm = widgets.ToggleButtons(
    options=['BatchNorm', 'InstanceNorm'],
    value=normval,
    description='Normalization:',
    disabled=False,
)

batch = widgets.ToggleButtons(
    options=['32', '64', '128'],
    value=batchval,
    description='Batch size:',
    disabled=False,
)

epochs = widgets.IntSlider(
    value=epochval,
    min=0,
    max=100,
    step=5,
    description='No. Epochs:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

lrate = widgets.FloatSlider(
    value=lrateval,
    min=0.000,
    max=0.001,
    step=0.0001,
    description='Learn rate:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.4f'
)

beta1 = widgets.FloatSlider(
    value=beta1val,
    min=0,
    max=1,
    step=0.1,
    description='Beta1:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f'
)

clamp = widgets.FloatSlider(
    value=clampval,
    min=0.00,
    max=0.10,
    step=0.01,
    description='Clamp*:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f'
)

savepath = widgets.Text(
    value=savepathval,
    placeholder='Past your path here',
    description='Save path:',
    disabled=False
)

def Show():
	display(Markdown('### Set model parameters:'))

	display(loss)
	display(optim)
	display(norm)
	display(batch)
	display(epochs)
	display(lrate)
	display(beta1)
	display(clamp)
	display(savepath)

	display(Markdown('\* Only relevant in case of use of Wasserstein distance.'))


def PrintModelSettings():
	display(Markdown('#### Please check your settings:'))
	print(GetFormattedSettings())

	display(Markdown('#### Results to be saved at:'))
	print(savepath.value)

def showrunmodelbutton():
	button = widgets.Button(description="Start Training*")
	button.on_click(run_all)
	display(button)
	print('* Runs all the cells below')

def run_all(ev):
    display(Javascript('IPython.notebook.execute_cells_below()'))


def GetFormattedSettings():
	modelsettings = 'ConvolutionalGAN, ' + loss.value + ' loss, ' + optim.value + ' optimizer, ' + norm.value + ', batchsize of ' + batch.value + ', learning rate of ' + str(lrate.value) + ', beta1 of ' + str(beta1.value) + ', for ' + str(epochs.value) + ' epochs.'
	
	if loss.value == 'Wasserstein':
		modelsettings += (' (Clamp value of ' + str(clamp.value) + ')')

	return modelsettings

