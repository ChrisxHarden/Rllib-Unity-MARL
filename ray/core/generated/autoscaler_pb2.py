# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/experimental/autoscaler.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n.src/ray/protobuf/experimental/autoscaler.proto\x12\x12ray.rpc.autoscaler\"X\n\x16\x41ntiAffinityConstraint\x12\x1d\n\nlabel_name\x18\x01 \x01(\tR\tlabelName\x12\x1f\n\x0blabel_value\x18\x02 \x01(\tR\nlabelValue\"T\n\x12\x41\x66\x66inityConstraint\x12\x1d\n\nlabel_name\x18\x01 \x01(\tR\tlabelName\x12\x1f\n\x0blabel_value\x18\x02 \x01(\tR\nlabelValue\"\xd3\x01\n\x13PlacementConstraint\x12T\n\ranti_affinity\x18\x01 \x01(\x0b\x32*.ray.rpc.autoscaler.AntiAffinityConstraintH\x00R\x0c\x61ntiAffinity\x88\x01\x01\x12G\n\x08\x61\x66\x66inity\x18\x02 \x01(\x0b\x32&.ray.rpc.autoscaler.AffinityConstraintH\x01R\x08\x61\x66\x66inity\x88\x01\x01\x42\x10\n\x0e_anti_affinityB\x0b\n\t_affinity\"\x98\x02\n\x0fResourceRequest\x12\x63\n\x10resources_bundle\x18\x01 \x03(\x0b\x32\x38.ray.rpc.autoscaler.ResourceRequest.ResourcesBundleEntryR\x0fresourcesBundle\x12\\\n\x15placement_constraints\x18\x02 \x03(\x0b\x32\'.ray.rpc.autoscaler.PlacementConstraintR\x14placementConstraints\x1a\x42\n\x14ResourcesBundleEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\"m\n\x16ResourceRequestByCount\x12=\n\x07request\x18\x01 \x01(\x0b\x32#.ray.rpc.autoscaler.ResourceRequestR\x07request\x12\x14\n\x05\x63ount\x18\x02 \x01(\x03R\x05\x63ount\"p\n\x13GangResourceRequest\x12?\n\x08requests\x18\x01 \x03(\x0b\x32#.ray.rpc.autoscaler.ResourceRequestR\x08requests\x12\x18\n\x07\x64\x65tails\x18\x02 \x01(\tR\x07\x64\x65tails\"h\n\x19\x43lusterResourceConstraint\x12K\n\x0bmin_bundles\x18\x01 \x03(\x0b\x32*.ray.rpc.autoscaler.ResourceRequestByCountR\nminBundles\"\xc1\x06\n\tNodeState\x12\x17\n\x07node_id\x18\x01 \x01(\x0cR\x06nodeId\x12\x1f\n\x0binstance_id\x18\x02 \x01(\tR\ninstanceId\x12+\n\x12ray_node_type_name\x18\x03 \x01(\tR\x0frayNodeTypeName\x12\x66\n\x13\x61vailable_resources\x18\x04 \x03(\x0b\x32\x35.ray.rpc.autoscaler.NodeState.AvailableResourcesEntryR\x12\x61vailableResources\x12Z\n\x0ftotal_resources\x18\x05 \x03(\x0b\x32\x31.ray.rpc.autoscaler.NodeState.TotalResourcesEntryR\x0etotalResources\x12W\n\x0e\x64ynamic_labels\x18\x06 \x03(\x0b\x32\x30.ray.rpc.autoscaler.NodeState.DynamicLabelsEntryR\rdynamicLabels\x12,\n\x12node_state_version\x18\x07 \x01(\x03R\x10nodeStateVersion\x12\x36\n\x06status\x18\x08 \x01(\x0e\x32\x1e.ray.rpc.autoscaler.NodeStatusR\x06status\x12(\n\x10idle_duration_ms\x18\t \x01(\x03R\x0eidleDurationMs\x12&\n\x0fnode_ip_address\x18\n \x01(\tR\rnodeIpAddress\x12,\n\x12instance_type_name\x18\x0b \x01(\tR\x10instanceTypeName\x1a\x45\n\x17\x41vailableResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\x1a\x41\n\x13TotalResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\x1a@\n\x12\x44ynamicLabelsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"w\n\x1eGetClusterResourceStateRequest\x12U\n(last_seen_cluster_resource_state_version\x18\x01 \x01(\x03R#lastSeenClusterResourceStateVersion\"\xe0\x04\n\x14\x43lusterResourceState\x12\x43\n\x1e\x63luster_resource_state_version\x18\x01 \x01(\x03R\x1b\x63lusterResourceStateVersion\x12J\n\"last_seen_autoscaler_state_version\x18\x02 \x01(\x03R\x1elastSeenAutoscalerStateVersion\x12>\n\x0bnode_states\x18\x03 \x03(\x0b\x32\x1d.ray.rpc.autoscaler.NodeStateR\nnodeStates\x12\x66\n\x19pending_resource_requests\x18\x04 \x03(\x0b\x32*.ray.rpc.autoscaler.ResourceRequestByCountR\x17pendingResourceRequests\x12l\n\x1epending_gang_resource_requests\x18\x05 \x03(\x0b\x32\'.ray.rpc.autoscaler.GangResourceRequestR\x1bpendingGangResourceRequests\x12o\n\x1c\x63luster_resource_constraints\x18\x06 \x03(\x0b\x32-.ray.rpc.autoscaler.ClusterResourceConstraintR\x1a\x63lusterResourceConstraints\x12\x30\n\x14\x63luster_session_name\x18\x07 \x01(\tR\x12\x63lusterSessionName\"~\n\x1cGetClusterResourceStateReply\x12^\n\x16\x63luster_resource_state\x18\x01 \x01(\x0b\x32(.ray.rpc.autoscaler.ClusterResourceStateR\x14\x63lusterResourceState\"\xa8\x01\n\x16PendingInstanceRequest\x12,\n\x12instance_type_name\x18\x01 \x01(\tR\x10instanceTypeName\x12+\n\x12ray_node_type_name\x18\x02 \x01(\tR\x0frayNodeTypeName\x12\x14\n\x05\x63ount\x18\x03 \x01(\x05R\x05\x63ount\x12\x1d\n\nrequest_ts\x18\x04 \x01(\x03R\trequestTs\"\xd8\x01\n\x15\x46\x61iledInstanceRequest\x12,\n\x12instance_type_name\x18\x01 \x01(\tR\x10instanceTypeName\x12+\n\x12ray_node_type_name\x18\x02 \x01(\tR\x0frayNodeTypeName\x12\x14\n\x05\x63ount\x18\x03 \x01(\x05R\x05\x63ount\x12\x16\n\x06reason\x18\x04 \x01(\tR\x06reason\x12\x19\n\x08start_ts\x18\x05 \x01(\x03R\x07startTs\x12\x1b\n\tfailed_ts\x18\x06 \x01(\x03R\x08\x66\x61iledTs\"\xc6\x01\n\x0fPendingInstance\x12,\n\x12instance_type_name\x18\x01 \x01(\tR\x10instanceTypeName\x12+\n\x12ray_node_type_name\x18\x02 \x01(\tR\x0frayNodeTypeName\x12\x1f\n\x0binstance_id\x18\x03 \x01(\tR\ninstanceId\x12\x1d\n\nip_address\x18\x04 \x01(\tR\tipAddress\x12\x18\n\x07\x64\x65tails\x18\x05 \x01(\tR\x07\x64\x65tails\"\xa4\x06\n\x10\x41utoscalingState\x12U\n(last_seen_cluster_resource_state_version\x18\x01 \x01(\x03R#lastSeenClusterResourceStateVersion\x12\x38\n\x18\x61utoscaler_state_version\x18\x02 \x01(\x03R\x16\x61utoscalerStateVersion\x12\x66\n\x19pending_instance_requests\x18\x03 \x03(\x0b\x32*.ray.rpc.autoscaler.PendingInstanceRequestR\x17pendingInstanceRequests\x12\x65\n\x1cinfeasible_resource_requests\x18\x04 \x03(\x0b\x32#.ray.rpc.autoscaler.ResourceRequestR\x1ainfeasibleResourceRequests\x12r\n!infeasible_gang_resource_requests\x18\x05 \x03(\x0b\x32\'.ray.rpc.autoscaler.GangResourceRequestR\x1einfeasibleGangResourceRequests\x12\x84\x01\n\'infeasible_cluster_resource_constraints\x18\x06 \x03(\x0b\x32-.ray.rpc.autoscaler.ClusterResourceConstraintR$infeasibleClusterResourceConstraints\x12P\n\x11pending_instances\x18\x07 \x03(\x0b\x32#.ray.rpc.autoscaler.PendingInstanceR\x10pendingInstances\x12\x63\n\x18\x66\x61iled_instance_requests\x18\x08 \x03(\x0b\x32).ray.rpc.autoscaler.FailedInstanceRequestR\x16\x66\x61iledInstanceRequests\"r\n\x1dReportAutoscalingStateRequest\x12Q\n\x11\x61utoscaling_state\x18\x01 \x01(\x0b\x32$.ray.rpc.autoscaler.AutoscalingStateR\x10\x61utoscalingState\"\x1d\n\x1bReportAutoscalingStateReply\"\x98\x01\n\'RequestClusterResourceConstraintRequest\x12m\n\x1b\x63luster_resource_constraint\x18\x01 \x01(\x0b\x32-.ray.rpc.autoscaler.ClusterResourceConstraintR\x19\x63lusterResourceConstraint\"\'\n%RequestClusterResourceConstraintReply\"\x19\n\x17GetClusterStatusRequest\"\xca\x01\n\x15GetClusterStatusReply\x12Q\n\x11\x61utoscaling_state\x18\x01 \x01(\x0b\x32$.ray.rpc.autoscaler.AutoscalingStateR\x10\x61utoscalingState\x12^\n\x16\x63luster_resource_state\x18\x02 \x01(\x0b\x32(.ray.rpc.autoscaler.ClusterResourceStateR\x14\x63lusterResourceState*>\n\nNodeStatus\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x0b\n\x07RUNNING\x10\x01\x12\x08\n\x04\x44\x45\x41\x44\x10\x02\x12\x08\n\x04IDLE\x10\x03\x32\xa0\x04\n\x16\x41utoscalerStateService\x12\x7f\n\x17GetClusterResourceState\x12\x32.ray.rpc.autoscaler.GetClusterResourceStateRequest\x1a\x30.ray.rpc.autoscaler.GetClusterResourceStateReply\x12|\n\x16ReportAutoscalingState\x12\x31.ray.rpc.autoscaler.ReportAutoscalingStateRequest\x1a/.ray.rpc.autoscaler.ReportAutoscalingStateReply\x12\x9a\x01\n RequestClusterResourceConstraint\x12;.ray.rpc.autoscaler.RequestClusterResourceConstraintRequest\x1a\x39.ray.rpc.autoscaler.RequestClusterResourceConstraintReply\x12j\n\x10GetClusterStatus\x12+.ray.rpc.autoscaler.GetClusterStatusRequest\x1a).ray.rpc.autoscaler.GetClusterStatusReplyB\x03\xf8\x01\x01\x62\x06proto3')

_NODESTATUS = DESCRIPTOR.enum_types_by_name['NodeStatus']
NodeStatus = enum_type_wrapper.EnumTypeWrapper(_NODESTATUS)
UNSPECIFIED = 0
RUNNING = 1
DEAD = 2
IDLE = 3


_ANTIAFFINITYCONSTRAINT = DESCRIPTOR.message_types_by_name['AntiAffinityConstraint']
_AFFINITYCONSTRAINT = DESCRIPTOR.message_types_by_name['AffinityConstraint']
_PLACEMENTCONSTRAINT = DESCRIPTOR.message_types_by_name['PlacementConstraint']
_RESOURCEREQUEST = DESCRIPTOR.message_types_by_name['ResourceRequest']
_RESOURCEREQUEST_RESOURCESBUNDLEENTRY = _RESOURCEREQUEST.nested_types_by_name['ResourcesBundleEntry']
_RESOURCEREQUESTBYCOUNT = DESCRIPTOR.message_types_by_name['ResourceRequestByCount']
_GANGRESOURCEREQUEST = DESCRIPTOR.message_types_by_name['GangResourceRequest']
_CLUSTERRESOURCECONSTRAINT = DESCRIPTOR.message_types_by_name['ClusterResourceConstraint']
_NODESTATE = DESCRIPTOR.message_types_by_name['NodeState']
_NODESTATE_AVAILABLERESOURCESENTRY = _NODESTATE.nested_types_by_name['AvailableResourcesEntry']
_NODESTATE_TOTALRESOURCESENTRY = _NODESTATE.nested_types_by_name['TotalResourcesEntry']
_NODESTATE_DYNAMICLABELSENTRY = _NODESTATE.nested_types_by_name['DynamicLabelsEntry']
_GETCLUSTERRESOURCESTATEREQUEST = DESCRIPTOR.message_types_by_name['GetClusterResourceStateRequest']
_CLUSTERRESOURCESTATE = DESCRIPTOR.message_types_by_name['ClusterResourceState']
_GETCLUSTERRESOURCESTATEREPLY = DESCRIPTOR.message_types_by_name['GetClusterResourceStateReply']
_PENDINGINSTANCEREQUEST = DESCRIPTOR.message_types_by_name['PendingInstanceRequest']
_FAILEDINSTANCEREQUEST = DESCRIPTOR.message_types_by_name['FailedInstanceRequest']
_PENDINGINSTANCE = DESCRIPTOR.message_types_by_name['PendingInstance']
_AUTOSCALINGSTATE = DESCRIPTOR.message_types_by_name['AutoscalingState']
_REPORTAUTOSCALINGSTATEREQUEST = DESCRIPTOR.message_types_by_name['ReportAutoscalingStateRequest']
_REPORTAUTOSCALINGSTATEREPLY = DESCRIPTOR.message_types_by_name['ReportAutoscalingStateReply']
_REQUESTCLUSTERRESOURCECONSTRAINTREQUEST = DESCRIPTOR.message_types_by_name['RequestClusterResourceConstraintRequest']
_REQUESTCLUSTERRESOURCECONSTRAINTREPLY = DESCRIPTOR.message_types_by_name['RequestClusterResourceConstraintReply']
_GETCLUSTERSTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetClusterStatusRequest']
_GETCLUSTERSTATUSREPLY = DESCRIPTOR.message_types_by_name['GetClusterStatusReply']
AntiAffinityConstraint = _reflection.GeneratedProtocolMessageType('AntiAffinityConstraint', (_message.Message,), {
  'DESCRIPTOR' : _ANTIAFFINITYCONSTRAINT,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.AntiAffinityConstraint)
  })
_sym_db.RegisterMessage(AntiAffinityConstraint)

AffinityConstraint = _reflection.GeneratedProtocolMessageType('AffinityConstraint', (_message.Message,), {
  'DESCRIPTOR' : _AFFINITYCONSTRAINT,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.AffinityConstraint)
  })
_sym_db.RegisterMessage(AffinityConstraint)

PlacementConstraint = _reflection.GeneratedProtocolMessageType('PlacementConstraint', (_message.Message,), {
  'DESCRIPTOR' : _PLACEMENTCONSTRAINT,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.PlacementConstraint)
  })
_sym_db.RegisterMessage(PlacementConstraint)

ResourceRequest = _reflection.GeneratedProtocolMessageType('ResourceRequest', (_message.Message,), {

  'ResourcesBundleEntry' : _reflection.GeneratedProtocolMessageType('ResourcesBundleEntry', (_message.Message,), {
    'DESCRIPTOR' : _RESOURCEREQUEST_RESOURCESBUNDLEENTRY,
    '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ResourceRequest.ResourcesBundleEntry)
    })
  ,
  'DESCRIPTOR' : _RESOURCEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ResourceRequest)
  })
_sym_db.RegisterMessage(ResourceRequest)
_sym_db.RegisterMessage(ResourceRequest.ResourcesBundleEntry)

ResourceRequestByCount = _reflection.GeneratedProtocolMessageType('ResourceRequestByCount', (_message.Message,), {
  'DESCRIPTOR' : _RESOURCEREQUESTBYCOUNT,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ResourceRequestByCount)
  })
_sym_db.RegisterMessage(ResourceRequestByCount)

GangResourceRequest = _reflection.GeneratedProtocolMessageType('GangResourceRequest', (_message.Message,), {
  'DESCRIPTOR' : _GANGRESOURCEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.GangResourceRequest)
  })
_sym_db.RegisterMessage(GangResourceRequest)

ClusterResourceConstraint = _reflection.GeneratedProtocolMessageType('ClusterResourceConstraint', (_message.Message,), {
  'DESCRIPTOR' : _CLUSTERRESOURCECONSTRAINT,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ClusterResourceConstraint)
  })
_sym_db.RegisterMessage(ClusterResourceConstraint)

NodeState = _reflection.GeneratedProtocolMessageType('NodeState', (_message.Message,), {

  'AvailableResourcesEntry' : _reflection.GeneratedProtocolMessageType('AvailableResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODESTATE_AVAILABLERESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.NodeState.AvailableResourcesEntry)
    })
  ,

  'TotalResourcesEntry' : _reflection.GeneratedProtocolMessageType('TotalResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODESTATE_TOTALRESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.NodeState.TotalResourcesEntry)
    })
  ,

  'DynamicLabelsEntry' : _reflection.GeneratedProtocolMessageType('DynamicLabelsEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODESTATE_DYNAMICLABELSENTRY,
    '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.NodeState.DynamicLabelsEntry)
    })
  ,
  'DESCRIPTOR' : _NODESTATE,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.NodeState)
  })
_sym_db.RegisterMessage(NodeState)
_sym_db.RegisterMessage(NodeState.AvailableResourcesEntry)
_sym_db.RegisterMessage(NodeState.TotalResourcesEntry)
_sym_db.RegisterMessage(NodeState.DynamicLabelsEntry)

GetClusterResourceStateRequest = _reflection.GeneratedProtocolMessageType('GetClusterResourceStateRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCLUSTERRESOURCESTATEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.GetClusterResourceStateRequest)
  })
_sym_db.RegisterMessage(GetClusterResourceStateRequest)

ClusterResourceState = _reflection.GeneratedProtocolMessageType('ClusterResourceState', (_message.Message,), {
  'DESCRIPTOR' : _CLUSTERRESOURCESTATE,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ClusterResourceState)
  })
_sym_db.RegisterMessage(ClusterResourceState)

GetClusterResourceStateReply = _reflection.GeneratedProtocolMessageType('GetClusterResourceStateReply', (_message.Message,), {
  'DESCRIPTOR' : _GETCLUSTERRESOURCESTATEREPLY,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.GetClusterResourceStateReply)
  })
_sym_db.RegisterMessage(GetClusterResourceStateReply)

PendingInstanceRequest = _reflection.GeneratedProtocolMessageType('PendingInstanceRequest', (_message.Message,), {
  'DESCRIPTOR' : _PENDINGINSTANCEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.PendingInstanceRequest)
  })
_sym_db.RegisterMessage(PendingInstanceRequest)

FailedInstanceRequest = _reflection.GeneratedProtocolMessageType('FailedInstanceRequest', (_message.Message,), {
  'DESCRIPTOR' : _FAILEDINSTANCEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.FailedInstanceRequest)
  })
_sym_db.RegisterMessage(FailedInstanceRequest)

PendingInstance = _reflection.GeneratedProtocolMessageType('PendingInstance', (_message.Message,), {
  'DESCRIPTOR' : _PENDINGINSTANCE,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.PendingInstance)
  })
_sym_db.RegisterMessage(PendingInstance)

AutoscalingState = _reflection.GeneratedProtocolMessageType('AutoscalingState', (_message.Message,), {
  'DESCRIPTOR' : _AUTOSCALINGSTATE,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.AutoscalingState)
  })
_sym_db.RegisterMessage(AutoscalingState)

ReportAutoscalingStateRequest = _reflection.GeneratedProtocolMessageType('ReportAutoscalingStateRequest', (_message.Message,), {
  'DESCRIPTOR' : _REPORTAUTOSCALINGSTATEREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ReportAutoscalingStateRequest)
  })
_sym_db.RegisterMessage(ReportAutoscalingStateRequest)

ReportAutoscalingStateReply = _reflection.GeneratedProtocolMessageType('ReportAutoscalingStateReply', (_message.Message,), {
  'DESCRIPTOR' : _REPORTAUTOSCALINGSTATEREPLY,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.ReportAutoscalingStateReply)
  })
_sym_db.RegisterMessage(ReportAutoscalingStateReply)

RequestClusterResourceConstraintRequest = _reflection.GeneratedProtocolMessageType('RequestClusterResourceConstraintRequest', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTCLUSTERRESOURCECONSTRAINTREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.RequestClusterResourceConstraintRequest)
  })
_sym_db.RegisterMessage(RequestClusterResourceConstraintRequest)

RequestClusterResourceConstraintReply = _reflection.GeneratedProtocolMessageType('RequestClusterResourceConstraintReply', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTCLUSTERRESOURCECONSTRAINTREPLY,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.RequestClusterResourceConstraintReply)
  })
_sym_db.RegisterMessage(RequestClusterResourceConstraintReply)

GetClusterStatusRequest = _reflection.GeneratedProtocolMessageType('GetClusterStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCLUSTERSTATUSREQUEST,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.GetClusterStatusRequest)
  })
_sym_db.RegisterMessage(GetClusterStatusRequest)

GetClusterStatusReply = _reflection.GeneratedProtocolMessageType('GetClusterStatusReply', (_message.Message,), {
  'DESCRIPTOR' : _GETCLUSTERSTATUSREPLY,
  '__module__' : 'src.ray.protobuf.experimental.autoscaler_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.autoscaler.GetClusterStatusReply)
  })
_sym_db.RegisterMessage(GetClusterStatusReply)

_AUTOSCALERSTATESERVICE = DESCRIPTOR.services_by_name['AutoscalerStateService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _RESOURCEREQUEST_RESOURCESBUNDLEENTRY._options = None
  _RESOURCEREQUEST_RESOURCESBUNDLEENTRY._serialized_options = b'8\001'
  _NODESTATE_AVAILABLERESOURCESENTRY._options = None
  _NODESTATE_AVAILABLERESOURCESENTRY._serialized_options = b'8\001'
  _NODESTATE_TOTALRESOURCESENTRY._options = None
  _NODESTATE_TOTALRESOURCESENTRY._serialized_options = b'8\001'
  _NODESTATE_DYNAMICLABELSENTRY._options = None
  _NODESTATE_DYNAMICLABELSENTRY._serialized_options = b'8\001'
  _NODESTATUS._serialized_start=4743
  _NODESTATUS._serialized_end=4805
  _ANTIAFFINITYCONSTRAINT._serialized_start=70
  _ANTIAFFINITYCONSTRAINT._serialized_end=158
  _AFFINITYCONSTRAINT._serialized_start=160
  _AFFINITYCONSTRAINT._serialized_end=244
  _PLACEMENTCONSTRAINT._serialized_start=247
  _PLACEMENTCONSTRAINT._serialized_end=458
  _RESOURCEREQUEST._serialized_start=461
  _RESOURCEREQUEST._serialized_end=741
  _RESOURCEREQUEST_RESOURCESBUNDLEENTRY._serialized_start=675
  _RESOURCEREQUEST_RESOURCESBUNDLEENTRY._serialized_end=741
  _RESOURCEREQUESTBYCOUNT._serialized_start=743
  _RESOURCEREQUESTBYCOUNT._serialized_end=852
  _GANGRESOURCEREQUEST._serialized_start=854
  _GANGRESOURCEREQUEST._serialized_end=966
  _CLUSTERRESOURCECONSTRAINT._serialized_start=968
  _CLUSTERRESOURCECONSTRAINT._serialized_end=1072
  _NODESTATE._serialized_start=1075
  _NODESTATE._serialized_end=1908
  _NODESTATE_AVAILABLERESOURCESENTRY._serialized_start=1706
  _NODESTATE_AVAILABLERESOURCESENTRY._serialized_end=1775
  _NODESTATE_TOTALRESOURCESENTRY._serialized_start=1777
  _NODESTATE_TOTALRESOURCESENTRY._serialized_end=1842
  _NODESTATE_DYNAMICLABELSENTRY._serialized_start=1844
  _NODESTATE_DYNAMICLABELSENTRY._serialized_end=1908
  _GETCLUSTERRESOURCESTATEREQUEST._serialized_start=1910
  _GETCLUSTERRESOURCESTATEREQUEST._serialized_end=2029
  _CLUSTERRESOURCESTATE._serialized_start=2032
  _CLUSTERRESOURCESTATE._serialized_end=2640
  _GETCLUSTERRESOURCESTATEREPLY._serialized_start=2642
  _GETCLUSTERRESOURCESTATEREPLY._serialized_end=2768
  _PENDINGINSTANCEREQUEST._serialized_start=2771
  _PENDINGINSTANCEREQUEST._serialized_end=2939
  _FAILEDINSTANCEREQUEST._serialized_start=2942
  _FAILEDINSTANCEREQUEST._serialized_end=3158
  _PENDINGINSTANCE._serialized_start=3161
  _PENDINGINSTANCE._serialized_end=3359
  _AUTOSCALINGSTATE._serialized_start=3362
  _AUTOSCALINGSTATE._serialized_end=4166
  _REPORTAUTOSCALINGSTATEREQUEST._serialized_start=4168
  _REPORTAUTOSCALINGSTATEREQUEST._serialized_end=4282
  _REPORTAUTOSCALINGSTATEREPLY._serialized_start=4284
  _REPORTAUTOSCALINGSTATEREPLY._serialized_end=4313
  _REQUESTCLUSTERRESOURCECONSTRAINTREQUEST._serialized_start=4316
  _REQUESTCLUSTERRESOURCECONSTRAINTREQUEST._serialized_end=4468
  _REQUESTCLUSTERRESOURCECONSTRAINTREPLY._serialized_start=4470
  _REQUESTCLUSTERRESOURCECONSTRAINTREPLY._serialized_end=4509
  _GETCLUSTERSTATUSREQUEST._serialized_start=4511
  _GETCLUSTERSTATUSREQUEST._serialized_end=4536
  _GETCLUSTERSTATUSREPLY._serialized_start=4539
  _GETCLUSTERSTATUSREPLY._serialized_end=4741
  _AUTOSCALERSTATESERVICE._serialized_start=4808
  _AUTOSCALERSTATESERVICE._serialized_end=5352
# @@protoc_insertion_point(module_scope)