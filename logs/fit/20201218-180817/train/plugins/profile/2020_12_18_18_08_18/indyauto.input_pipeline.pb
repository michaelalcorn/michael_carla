	����v��?����v��?!����v��?	�E��G#@�E��G#@!�E��G#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$����v��?7�ُ��?A���}r�?Y*ʥ���?*	/�$!Z@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate C�*�?!��}�	P@)�H��_��?1E��ԙ�N@:Preprocessing2F
Iterator::Modelwd�6���?!��FЖ�8@)6�
�r�?1�T��o-@:Preprocessing2U
Iterator::Model::ParallelMapV2��M~�N�?!
���$@)��M~�N�?1
���$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF
e��k}?!�I:�}@)'0��mp?1�T>Ɩ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm�i�*�i?!��S��G@)m�i�*�i?1��S��G@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipΉ=���?!CS�K�R@)������i?1�?�u�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[1]::FromTensor����y]?!>���O��?)����y]?1>���O��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I�p�?!�O��|KP@)"��u��Q?1�rn��t�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[6]::Concatenate[0]::TensorSlice�x#��??!�8����?)�x#��??1�8����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t16.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�E��G#@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	7�ُ��?7�ُ��?!7�ُ��?      ��!       "      ��!       *      ��!       2	���}r�?���}r�?!���}r�?:      ��!       B      ��!       J	*ʥ���?*ʥ���?!*ʥ���?R      ��!       Z	*ʥ���?*ʥ���?!*ʥ���?JCPU_ONLYY�E��G#@b 