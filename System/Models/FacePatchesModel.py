import numpy as np
import tensorflow as tf

MAPPING = {0:'neutral', 1:'anger', 2:'surprise', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness'}
MP = './Models/model_weights/'
DEFAULT_PADDING = 'SAME'
TypeThreshold=100
eye_p_shape=[None, 26, 64, 1]
midd_p_shape=[None, 49, 28, 1]
mou_p_shape=[None, 30, 54, 1]
            


###dependent modules for network definition
def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated
class Network(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        #self._Variables = []#use to store the variables' names created
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def var_summaries(self, var, name):
        '''add summaries information for the var (basically for weights) at the TensorBoard visualization'''
        with tf.name_scope(name+'_summaries'):
            mean=tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram',var)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        #tfv=tf.get_variable(name, shape, trainable=self.trainable)
        #self.var_summaries(var=tfv, name=name)#this will slow down the training significantly
        #return tfv
        return tf.get_variable(name, shape, trainable=self.trainable)
        #return tf.Variable(tf.random_normal(shape),dtype=tf.float32,name=name,trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, (c_i//group), c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def conv1D(self,
             input,
             k_s,
             c_o,
             stride,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve1D = lambda i, k: tf.nn.conv1d(i, k, stride, padding=padding)
        #convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_s, (c_i//group), c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve1D(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve1D(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            elif input_shape.ndims == 3:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = list(map(lambda v: v.value, input.get_shape()))
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, scale_offset=False, relu=False):##default scale_offset is True
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def hiddenLayer(self,
             input,
             c_o,
             name,
             relu=False,
             activeF=False,
             act_func=tf.nn.sigmoid):
        '''Layer for 1D features'''
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        # Verify that the grouping parameter is valid
        with tf.variable_scope(name) as scope:
            weights = self.make_var('weights',[c_i,c_o])
            output=tf.matmul(input, weights)
            # Add the biases
            biases = self.make_var('bias',[c_o])
            if activeF:
               output = act_func(output+biases)
            else:
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
###
#3 network definition
class FacePatches_NET_3Conv_2Inception(Network):
    def setup(self):
        (self.feed('eyePatch_data')#64*26
             .conv(3, 3, 8, 1, 1, name='eye_conv1_1_3x3')#64*26*16
             .conv(3, 3, 8, 1, 1, name='eye_conv1_2_3x3')#64*26*16
             .max_pool(2, 2, 2, 2, name='eye_pool1')#32*13*16
             .conv(3, 3, 32, 1, 1, name='eye_conv2_1_3x3')#32*13*64
             .conv(3, 3, 32, 1, 1, name='eye_conv2_2_3x3')#
             .max_pool(2, 2, 2, 2, name='eye_pool2')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'eye_pool3')
             .fc(1024, name='eye_fc1'))
        (self.feed('eye_pool2')
             .fc(1024, name = 'eye_fc2'))
          

        (self.feed('middlePatch_data')#28*49
             .conv(3, 3, 8, 1, 1, name='middle_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='middle_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='middle_pool1')#
             .conv(3, 3, 32, 1, 1, name='middle_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='middle_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='middle_pool2')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'middle_pool3')
             .fc(1024, name='middle_fc1'))#
        (self.feed('middle_pool2')
             .fc(1024, name = 'middle_fc2'))

        (self.feed('mouthPatch_data')#54*30
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='mouth_pool1')#
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='mouth_pool2')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'mouth_pool3')
             .fc(1024, name='mouth_fc1'))#
        (self.feed('mouth_pool2')
             .fc(1024, name = 'mouth_fc2'))

        (self.feed('eye_fc1',
                   'eye_fc2',
                   'middle_fc1',
                   'middle_fc2',
                   'mouth_fc1',
                   'mouth_fc2')
             .concat(1, name='fusion_1')
             .fc(2048, name='fc1')
             .dropout(0.5, name='drop1')
             .fc(2048, name = 'fc2')
             .dropout(0.5, name='drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name = 'prob'))
#4 network definition
class FacePatches_NET_3Conv_1Inception(Network):
    def setup(self):
        (self.feed('eyePatch_data')#64*26
             .conv(3, 3, 8, 1, 1, name='eye_conv1_1_3x3')#64*26*16
             .conv(3, 3, 8, 1, 1, name='eye_conv1_2_3x3')#64*26*16
             .max_pool(2, 2, 2, 2, name='eye_pool1')#32*13*16
             .conv(3, 3, 32, 1, 1, name='eye_conv2_1_3x3')#32*13*64
             .conv(3, 3, 32, 1, 1, name='eye_conv2_2_3x3')#
             .max_pool(2, 2, 2, 2, name='eye_pool2')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'eye_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'eye_pool3')
             .fc(1024, name='eye_fc1'))
        #(self.feed('eye_pool2')
        #     .fc(1024, name = 'eye_fc2'))

        (self.feed('middlePatch_data')#28*49
             .conv(3, 3, 8, 1, 1, name='middle_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='middle_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='middle_pool1')#
             .conv(3, 3, 32, 1, 1, name='middle_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='middle_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='middle_pool2')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'middle_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'middle_pool3')
             .fc(1024, name='middle_fc1'))#
        #(self.feed('middle_pool2')
        #     .fc(1024, name = 'middle_fc2'))

        (self.feed('mouthPatch_data')#54*30
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_1_3x3')#
             .conv(3, 3, 8, 1, 1, name='mouth_conv1_2_3x3')#
             .max_pool(2, 2, 2, 2, name='mouth_pool1')#
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_1_3x3')
             .conv(3, 3, 32, 1, 1, name='mouth_conv2_2_3x3')
             .max_pool(2, 2, 2, 2, name='mouth_pool2')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_1_3x3')
             .conv(3, 3, 128, 1, 1, name = 'mouth_conv3_2_3x3')
             .max_pool(2, 2, 2, 2, name = 'mouth_pool3')
             .fc(1024, name='mouth_fc1'))#
        #(self.feed('mouth_pool2')
        #     .fc(1024, name = 'mouth_fc2'))

        (self.feed('eye_fc1',
                   'middle_fc1',
                   'mouth_fc1')
             .concat(1, name='fusion_1')
             .fc(2048, name='fc1')
             .dropout(0.5, name='drop1')
             .fc(2048, name = 'fc2')
             .dropout(0.5, name='drop2')
             .fc(7, relu = False, name = 'fc7')
             .softmax(name = 'prob'))

#
def getModelPathForPrediction(mid=0):
    if mid==300:
        mp=MP+'D502_M3_N3_T0_V0_R4_20171009235521_1.1895357370_.ckpt-16197'#0.9587	

    elif mid==301:
        mp=MP+'D502_M3_N3_T4_V4_R4_20171010084104_1.2033878565_.ckpt-18110'#0.9165

    elif mid==303:
        mp=MP+'D502_M3_N3_T5_V5_R4_20171010103653_1.1808838844_.ckpt-19024'#0.9779
    
    else:
        print('Unexpected Model ID. TRY another one.')
        exit(-1)
    return mp
#model for prediction
class FacePatchesModel:
    def __init__(self, mid=300):
        ###define the graph
        self.networkGraph=tf.Graph()
        with self.networkGraph.as_default():
            self.eye_p = tf.placeholder(tf.float32, eye_p_shape)
            self.midd_p = tf.placeholder(tf.float32, midd_p_shape)
            self.mou_p = tf.placeholder(tf.float32, mou_p_shape)
            if (mid//TypeThreshold)==3:
                self.network = FacePatches_NET_3Conv_2Inception({'eyePatch_data':self.eye_p,
                                                                'middlePatch_data':self.midd_p,
                                                                'mouthPatch_data':self.mou_p})
            elif (mid//TypeThreshold)==4:
                self.network = FacePatches_NET_3Conv_1Inception({'eyePatch_data':self.eye_p,
                                                                'middlePatch_data':self.midd_p,
                                                                'mouthPatch_data':self.mou_p})
            else:
                print('ERROR: Unexpected network type. Try another mid')
                exit(-1)
            self.saver=tf.train.Saver()
            self.prob=self.network.layers['prob']

            ###load pretrained model
            self.sess=tf.InteractiveSession(graph=self.networkGraph)
            try:
                #must initialize the variables in the graph for compution or loading pretrained weights
                self.sess.run(tf.variables_initializer(var_list=self.networkGraph.get_collection(name='variables')))
                print('Network variables initialized.')
                #the saver must define in the graph of its owner session, or it will occur error in restoration or saving
                self.saver.restore(sess=self.sess, save_path=getModelPathForPrediction(mid))
                print('Network Model loaded\n')
            except:
                print('ERROR: Unable to load the pretrained network.')
                exit(2)

    def predict(self, eye_p, midd_p, mou_p):
        probability = self.prob.eval(feed_dict={self.eye_p:eye_p, self.midd_p:midd_p, self.mou_p:mou_p})
        emotion = [MAPPING[np.argmax(pro)] for pro in probability]
        return emotion, probability
