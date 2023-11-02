# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/monitor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1esrc/ray/protobuf/monitor.proto\x12\x07ray.rpc\"\x16\n\x14GetRayVersionRequest\".\n\x12GetRayVersionReply\x12\x18\n\x07version\x18\x01 \x01(\tR\x07version\"4\n\x17\x44rainAndKillNodeRequest\x12\x19\n\x08node_ids\x18\x01 \x03(\x0cR\x07nodeIds\"<\n\x15\x44rainAndKillNodeReply\x12#\n\rdrained_nodes\x18\x02 \x03(\x0cR\x0c\x64rainedNodes\"\x94\x01\n\x0eResourceBundle\x12\x44\n\tresources\x18\x01 \x03(\x0b\x32&.ray.rpc.ResourceBundle.ResourcesEntryR\tresources\x1a<\n\x0eResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\"\xe7\x02\n\x0fResourceRequest\x12`\n\x15resource_request_type\x18\x01 \x01(\x0e\x32,.ray.rpc.ResourceRequest.ResourceRequestTypeR\x13resourceRequestType\x12\x14\n\x05\x63ount\x18\x02 \x01(\x05R\x05\x63ount\x12\x31\n\x07\x62undles\x18\x03 \x03(\x0b\x32\x17.ray.rpc.ResourceBundleR\x07\x62undles\"\xa8\x01\n\x13ResourceRequestType\x12\x14\n\x10TASK_RESERVATION\x10\x00\x12\x1d\n\x19STRICT_SPREAD_RESERVATION\x10\x01\x12\x16\n\x12SPREAD_RESERVATION\x10\x02\x12\x14\n\x10PACK_RESERVATION\x10\x03\x12\x1b\n\x17STRICT_PACK_RESERVATION\x10\x04\x12\x11\n\rMIN_RESOURCES\x10\x05\"\xf9\x02\n\nNodeStatus\x12\x17\n\x07node_id\x18\x01 \x01(\x0cR\x06nodeId\x12\x18\n\x07\x61\x64\x64ress\x18\x02 \x01(\tR\x07\x61\x64\x64ress\x12\\\n\x13\x61vailable_resources\x18\x03 \x03(\x0b\x32+.ray.rpc.NodeStatus.AvailableResourcesEntryR\x12\x61vailableResources\x12P\n\x0ftotal_resources\x18\x04 \x03(\x0b\x32\'.ray.rpc.NodeStatus.TotalResourcesEntryR\x0etotalResources\x1a\x45\n\x17\x41vailableResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\x1a\x41\n\x13TotalResourcesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\"\x1c\n\x1aGetSchedulingStatusRequest\"\x9b\x01\n\x18GetSchedulingStatusReply\x12\x45\n\x11resource_requests\x18\x01 \x03(\x0b\x32\x18.ray.rpc.ResourceRequestR\x10resourceRequests\x12\x38\n\rnode_statuses\x18\x02 \x03(\x0b\x32\x13.ray.rpc.NodeStatusR\x0cnodeStatuses2\x95\x02\n\x11MonitorGcsService\x12K\n\rGetRayVersion\x12\x1d.ray.rpc.GetRayVersionRequest\x1a\x1b.ray.rpc.GetRayVersionReply\x12T\n\x10\x44rainAndKillNode\x12 .ray.rpc.DrainAndKillNodeRequest\x1a\x1e.ray.rpc.DrainAndKillNodeReply\x12]\n\x13GetSchedulingStatus\x12#.ray.rpc.GetSchedulingStatusRequest\x1a!.ray.rpc.GetSchedulingStatusReplyB\x03\xf8\x01\x01\x62\x06proto3')



_GETRAYVERSIONREQUEST = DESCRIPTOR.message_types_by_name['GetRayVersionRequest']
_GETRAYVERSIONREPLY = DESCRIPTOR.message_types_by_name['GetRayVersionReply']
_DRAINANDKILLNODEREQUEST = DESCRIPTOR.message_types_by_name['DrainAndKillNodeRequest']
_DRAINANDKILLNODEREPLY = DESCRIPTOR.message_types_by_name['DrainAndKillNodeReply']
_RESOURCEBUNDLE = DESCRIPTOR.message_types_by_name['ResourceBundle']
_RESOURCEBUNDLE_RESOURCESENTRY = _RESOURCEBUNDLE.nested_types_by_name['ResourcesEntry']
_RESOURCEREQUEST = DESCRIPTOR.message_types_by_name['ResourceRequest']
_NODESTATUS = DESCRIPTOR.message_types_by_name['NodeStatus']
_NODESTATUS_AVAILABLERESOURCESENTRY = _NODESTATUS.nested_types_by_name['AvailableResourcesEntry']
_NODESTATUS_TOTALRESOURCESENTRY = _NODESTATUS.nested_types_by_name['TotalResourcesEntry']
_GETSCHEDULINGSTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetSchedulingStatusRequest']
_GETSCHEDULINGSTATUSREPLY = DESCRIPTOR.message_types_by_name['GetSchedulingStatusReply']
_RESOURCEREQUEST_RESOURCEREQUESTTYPE = _RESOURCEREQUEST.enum_types_by_name['ResourceRequestType']
GetRayVersionRequest = _reflection.GeneratedProtocolMessageType('GetRayVersionRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETRAYVERSIONREQUEST,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetRayVersionRequest)
  })
_sym_db.RegisterMessage(GetRayVersionRequest)

GetRayVersionReply = _reflection.GeneratedProtocolMessageType('GetRayVersionReply', (_message.Message,), {
  'DESCRIPTOR' : _GETRAYVERSIONREPLY,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetRayVersionReply)
  })
_sym_db.RegisterMessage(GetRayVersionReply)

DrainAndKillNodeRequest = _reflection.GeneratedProtocolMessageType('DrainAndKillNodeRequest', (_message.Message,), {
  'DESCRIPTOR' : _DRAINANDKILLNODEREQUEST,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DrainAndKillNodeRequest)
  })
_sym_db.RegisterMessage(DrainAndKillNodeRequest)

DrainAndKillNodeReply = _reflection.GeneratedProtocolMessageType('DrainAndKillNodeReply', (_message.Message,), {
  'DESCRIPTOR' : _DRAINANDKILLNODEREPLY,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DrainAndKillNodeReply)
  })
_sym_db.RegisterMessage(DrainAndKillNodeReply)

ResourceBundle = _reflection.GeneratedProtocolMessageType('ResourceBundle', (_message.Message,), {

  'ResourcesEntry' : _reflection.GeneratedProtocolMessageType('ResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _RESOURCEBUNDLE_RESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.monitor_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ResourceBundle.ResourcesEntry)
    })
  ,
  'DESCRIPTOR' : _RESOURCEBUNDLE,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ResourceBundle)
  })
_sym_db.RegisterMessage(ResourceBundle)
_sym_db.RegisterMessage(ResourceBundle.ResourcesEntry)

ResourceRequest = _reflection.GeneratedProtocolMessageType('ResourceRequest', (_message.Message,), {
  'DESCRIPTOR' : _RESOURCEREQUEST,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ResourceRequest)
  })
_sym_db.RegisterMessage(ResourceRequest)

NodeStatus = _reflection.GeneratedProtocolMessageType('NodeStatus', (_message.Message,), {

  'AvailableResourcesEntry' : _reflection.GeneratedProtocolMessageType('AvailableResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODESTATUS_AVAILABLERESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.monitor_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.NodeStatus.AvailableResourcesEntry)
    })
  ,

  'TotalResourcesEntry' : _reflection.GeneratedProtocolMessageType('TotalResourcesEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODESTATUS_TOTALRESOURCESENTRY,
    '__module__' : 'src.ray.protobuf.monitor_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.NodeStatus.TotalResourcesEntry)
    })
  ,
  'DESCRIPTOR' : _NODESTATUS,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.NodeStatus)
  })
_sym_db.RegisterMessage(NodeStatus)
_sym_db.RegisterMessage(NodeStatus.AvailableResourcesEntry)
_sym_db.RegisterMessage(NodeStatus.TotalResourcesEntry)

GetSchedulingStatusRequest = _reflection.GeneratedProtocolMessageType('GetSchedulingStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETSCHEDULINGSTATUSREQUEST,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetSchedulingStatusRequest)
  })
_sym_db.RegisterMessage(GetSchedulingStatusRequest)

GetSchedulingStatusReply = _reflection.GeneratedProtocolMessageType('GetSchedulingStatusReply', (_message.Message,), {
  'DESCRIPTOR' : _GETSCHEDULINGSTATUSREPLY,
  '__module__' : 'src.ray.protobuf.monitor_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetSchedulingStatusReply)
  })
_sym_db.RegisterMessage(GetSchedulingStatusReply)

_MONITORGCSSERVICE = DESCRIPTOR.services_by_name['MonitorGcsService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _RESOURCEBUNDLE_RESOURCESENTRY._options = None
  _RESOURCEBUNDLE_RESOURCESENTRY._serialized_options = b'8\001'
  _NODESTATUS_AVAILABLERESOURCESENTRY._options = None
  _NODESTATUS_AVAILABLERESOURCESENTRY._serialized_options = b'8\001'
  _NODESTATUS_TOTALRESOURCESENTRY._options = None
  _NODESTATUS_TOTALRESOURCESENTRY._serialized_options = b'8\001'
  _GETRAYVERSIONREQUEST._serialized_start=43
  _GETRAYVERSIONREQUEST._serialized_end=65
  _GETRAYVERSIONREPLY._serialized_start=67
  _GETRAYVERSIONREPLY._serialized_end=113
  _DRAINANDKILLNODEREQUEST._serialized_start=115
  _DRAINANDKILLNODEREQUEST._serialized_end=167
  _DRAINANDKILLNODEREPLY._serialized_start=169
  _DRAINANDKILLNODEREPLY._serialized_end=229
  _RESOURCEBUNDLE._serialized_start=232
  _RESOURCEBUNDLE._serialized_end=380
  _RESOURCEBUNDLE_RESOURCESENTRY._serialized_start=320
  _RESOURCEBUNDLE_RESOURCESENTRY._serialized_end=380
  _RESOURCEREQUEST._serialized_start=383
  _RESOURCEREQUEST._serialized_end=742
  _RESOURCEREQUEST_RESOURCEREQUESTTYPE._serialized_start=574
  _RESOURCEREQUEST_RESOURCEREQUESTTYPE._serialized_end=742
  _NODESTATUS._serialized_start=745
  _NODESTATUS._serialized_end=1122
  _NODESTATUS_AVAILABLERESOURCESENTRY._serialized_start=986
  _NODESTATUS_AVAILABLERESOURCESENTRY._serialized_end=1055
  _NODESTATUS_TOTALRESOURCESENTRY._serialized_start=1057
  _NODESTATUS_TOTALRESOURCESENTRY._serialized_end=1122
  _GETSCHEDULINGSTATUSREQUEST._serialized_start=1124
  _GETSCHEDULINGSTATUSREQUEST._serialized_end=1152
  _GETSCHEDULINGSTATUSREPLY._serialized_start=1155
  _GETSCHEDULINGSTATUSREPLY._serialized_end=1310
  _MONITORGCSSERVICE._serialized_start=1313
  _MONITORGCSSERVICE._serialized_end=1590
# @@protoc_insertion_point(module_scope)
