	�0�q��?�0�q��?!�0�q��?	����]�"@����]�"@!����]�"@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�0�q��?�dU����?ACV�zN�?Y�E(����?*	�|?5^:M@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�O��?!=m�۶@@)!�bG�P�?1xf�{(:@:Preprocessing2F
Iterator::Model��G�C��?!�����C@)c�~�x�?1���(8@:Preprocessing2U
Iterator::Model::ParallelMapV2(b�c�?!}��6��.@)(b�c�?1}��6��.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�L/1��?!{�W�3@)rѬl?1�����?*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�� �rhq?!t�@)�� �rhq?1t�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT�^P�?!c9>N@)����p?1-Q9T@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicew�ِfp?!�b��e@)w�ِfp?1�b��e@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t16.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9����]�"@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�dU����?�dU����?!�dU����?      ��!       "      ��!       *      ��!       2	CV�zN�?CV�zN�?!CV�zN�?:      ��!       B      ��!       J	�E(����?�E(����?!�E(����?R      ��!       Z	�E(����?�E(����?!�E(����?JCPU_ONLYY����]�"@b 