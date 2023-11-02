# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/usage.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1csrc/ray/protobuf/usage.proto\x12\tray.usage*\xae\x0c\n\x06TagKey\x12\n\n\x06_TEST1\x10\x00\x12\n\n\x06_TEST2\x10\x01\x12\x13\n\x0fRLLIB_FRAMEWORK\x10\x02\x12\x13\n\x0fRLLIB_ALGORITHM\x10\x03\x12\x15\n\x11RLLIB_NUM_WORKERS\x10\x04\x12\x15\n\x11SERVE_API_VERSION\x10\x05\x12\x19\n\x15SERVE_NUM_DEPLOYMENTS\x10\x06\x12\x0f\n\x0bGCS_STORAGE\x10\x07\x12\x1d\n\x19SERVE_NUM_GPU_DEPLOYMENTS\x10\x08\x12\x16\n\x12SERVE_FASTAPI_USED\x10\t\x12\x19\n\x15SERVE_DAG_DRIVER_USED\x10\n\x12\x1b\n\x17SERVE_HTTP_ADAPTER_USED\x10\x0b\x12\x1b\n\x17SERVE_GRPC_INGRESS_USED\x10\x0c\x12\x1a\n\x16SERVE_REST_API_VERSION\x10\r\x12\x12\n\x0eSERVE_NUM_APPS\x10\x0e\x12*\n&SERVE_NUM_REPLICAS_LIGHTWEIGHT_UPDATED\x10\x0f\x12)\n%SERVE_USER_CONFIG_LIGHTWEIGHT_UPDATED\x10\x10\x12\x30\n,SERVE_AUTOSCALING_CONFIG_LIGHTWEIGHT_UPDATED\x10\x11\x12\x1e\n\x1a\x43ORE_STATE_API_LIST_ACTORS\x10\x64\x12\x1d\n\x19\x43ORE_STATE_API_LIST_TASKS\x10\x65\x12\x1c\n\x18\x43ORE_STATE_API_LIST_JOBS\x10\x66\x12\x1d\n\x19\x43ORE_STATE_API_LIST_NODES\x10g\x12(\n$CORE_STATE_API_LIST_PLACEMENT_GROUPS\x10h\x12\x1f\n\x1b\x43ORE_STATE_API_LIST_WORKERS\x10i\x12\x1f\n\x1b\x43ORE_STATE_API_LIST_OBJECTS\x10j\x12$\n CORE_STATE_API_LIST_RUNTIME_ENVS\x10k\x12&\n\"CORE_STATE_API_LIST_CLUSTER_EVENTS\x10l\x12\x1c\n\x18\x43ORE_STATE_API_LIST_LOGS\x10m\x12\x1a\n\x16\x43ORE_STATE_API_GET_LOG\x10n\x12\"\n\x1e\x43ORE_STATE_API_SUMMARIZE_TASKS\x10o\x12#\n\x1f\x43ORE_STATE_API_SUMMARIZE_ACTORS\x10p\x12$\n CORE_STATE_API_SUMMARIZE_OBJECTS\x10q\x12\x13\n\x0e\x44\x41SHBOARD_USED\x10\xc8\x01\x12)\n$DASHBOARD_METRICS_PROMETHEUS_ENABLED\x10\xc9\x01\x12&\n!DASHBOARD_METRICS_GRAFANA_ENABLED\x10\xca\x01\x12\x13\n\x0ePG_NUM_CREATED\x10\xac\x02\x12\x16\n\x11\x41\x43TOR_NUM_CREATED\x10\xad\x02\x12\x1e\n\x19WORKER_CRASH_SYSTEM_ERROR\x10\xae\x02\x12\x15\n\x10WORKER_CRASH_OOM\x10\xaf\x02\x12\x19\n\x14RAY_GET_TIMEOUT_ZERO\x10\xb0\x02\x12\x1d\n\x18NUM_ACTOR_CREATION_TASKS\x10\xb1\x02\x12\x14\n\x0fNUM_ACTOR_TASKS\x10\xb2\x02\x12\x15\n\x10NUM_NORMAL_TASKS\x10\xb3\x02\x12\x10\n\x0bNUM_DRIVERS\x10\xb4\x02\x12\"\n\x1d\x45XPERIMENTAL_STATE_API_IMPORT\x10\xb5\x02\x12\x15\n\x10\x44\x41TA_LOGICAL_OPS\x10\x90\x03\x12\x10\n\x0b\x41IR_TRAINER\x10\xf4\x03\x12\x12\n\rTUNE_SEARCHER\x10\xf5\x03\x12\x13\n\x0eTUNE_SCHEDULER\x10\xf6\x03\x12\x11\n\x0c\x41IR_ENV_VARS\x10\xf7\x03\x12%\n AIR_SETUP_WANDB_INTEGRATION_USED\x10\xf8\x03\x12&\n!AIR_SETUP_MLFLOW_INTEGRATION_USED\x10\xf9\x03\x12\x12\n\rAIR_CALLBACKS\x10\xfa\x03\x12\x1e\n\x19\x41IR_STORAGE_CONFIGURATION\x10\xfb\x03\x12\x13\n\x0e\x41IR_ENTRYPOINT\x10\xfc\x03\x42\x03\xf8\x01\x01\x62\x06proto3')

_TAGKEY = DESCRIPTOR.enum_types_by_name['TagKey']
TagKey = enum_type_wrapper.EnumTypeWrapper(_TAGKEY)
_TEST1 = 0
_TEST2 = 1
RLLIB_FRAMEWORK = 2
RLLIB_ALGORITHM = 3
RLLIB_NUM_WORKERS = 4
SERVE_API_VERSION = 5
SERVE_NUM_DEPLOYMENTS = 6
GCS_STORAGE = 7
SERVE_NUM_GPU_DEPLOYMENTS = 8
SERVE_FASTAPI_USED = 9
SERVE_DAG_DRIVER_USED = 10
SERVE_HTTP_ADAPTER_USED = 11
SERVE_GRPC_INGRESS_USED = 12
SERVE_REST_API_VERSION = 13
SERVE_NUM_APPS = 14
SERVE_NUM_REPLICAS_LIGHTWEIGHT_UPDATED = 15
SERVE_USER_CONFIG_LIGHTWEIGHT_UPDATED = 16
SERVE_AUTOSCALING_CONFIG_LIGHTWEIGHT_UPDATED = 17
CORE_STATE_API_LIST_ACTORS = 100
CORE_STATE_API_LIST_TASKS = 101
CORE_STATE_API_LIST_JOBS = 102
CORE_STATE_API_LIST_NODES = 103
CORE_STATE_API_LIST_PLACEMENT_GROUPS = 104
CORE_STATE_API_LIST_WORKERS = 105
CORE_STATE_API_LIST_OBJECTS = 106
CORE_STATE_API_LIST_RUNTIME_ENVS = 107
CORE_STATE_API_LIST_CLUSTER_EVENTS = 108
CORE_STATE_API_LIST_LOGS = 109
CORE_STATE_API_GET_LOG = 110
CORE_STATE_API_SUMMARIZE_TASKS = 111
CORE_STATE_API_SUMMARIZE_ACTORS = 112
CORE_STATE_API_SUMMARIZE_OBJECTS = 113
DASHBOARD_USED = 200
DASHBOARD_METRICS_PROMETHEUS_ENABLED = 201
DASHBOARD_METRICS_GRAFANA_ENABLED = 202
PG_NUM_CREATED = 300
ACTOR_NUM_CREATED = 301
WORKER_CRASH_SYSTEM_ERROR = 302
WORKER_CRASH_OOM = 303
RAY_GET_TIMEOUT_ZERO = 304
NUM_ACTOR_CREATION_TASKS = 305
NUM_ACTOR_TASKS = 306
NUM_NORMAL_TASKS = 307
NUM_DRIVERS = 308
EXPERIMENTAL_STATE_API_IMPORT = 309
DATA_LOGICAL_OPS = 400
AIR_TRAINER = 500
TUNE_SEARCHER = 501
TUNE_SCHEDULER = 502
AIR_ENV_VARS = 503
AIR_SETUP_WANDB_INTEGRATION_USED = 504
AIR_SETUP_MLFLOW_INTEGRATION_USED = 505
AIR_CALLBACKS = 506
AIR_STORAGE_CONFIGURATION = 507
AIR_ENTRYPOINT = 508


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _TAGKEY._serialized_start=44
  _TAGKEY._serialized_end=1626
# @@protoc_insertion_point(module_scope)
