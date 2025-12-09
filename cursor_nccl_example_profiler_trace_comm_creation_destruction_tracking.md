# NCCL point-to-point profiler trace analysis
_Exported on 12/8/2025 at 21:05:01 GMT+1 from Cursor (2.0.77)_

---

**User**

[



{"name": "Group API", "cat": "GROUP_API", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1467846.533691, "args": {"groupApiId": 0, "groupDepth":2}},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"groupId": 0, "Stream": 0x262ca50}},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 1804044.097656},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "b", "id": 1, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"groupId": 0, "Stream": 0x420aa70}},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "e", "id": 1, "pid": 3937375, "tid": 1, "ts": 1808854.597656},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "b", "id": 2, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"groupId": 0, "Stream": 0x4142e50}},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "e", "id": 2, "pid": 3937375, "tid": 1, "ts": 1813821.208008},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "b", "id": 3, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"groupId": 0, "Stream": 0x4142e90}},

{"name": "KernelLaunch", "cat": "KERNEL_LAUNCH", "ph": "e", "id": 3, "pid": 3937375, "tid": 1, "ts": 1818540.877930},

{"name": "Send", "cat": "P2P_API", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x4142e90}},

{"name": "Send", "cat": "P2P", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1818542.097656, "args": {"CommHash": 12715772615376301750, "Rank": 0, "Peer": 1, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824045.316895, "args": {"Channel": 16, "StartGpuClk": 1765223116927674080, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824045.666992, "args": {"Channel": 17, "StartGpuClk": 1765223116927674080, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824046.106934, "args": {"Channel": 18, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824046.386719, "args": {"Channel": 19, "StartGpuClk": 1765223116927674112, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824046.946777, "args": {"Channel": 20, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.216797, "args": {"Channel": 21, "StartGpuClk": 1765223116927674144, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.486816, "args": {"Channel": 22, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.656738, "args": {"Channel": 23, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.796875, "args": {"Channel": 24, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.116699, "args": {"Channel": 25, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.386719, "args": {"Channel": 26, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.476562, "args": {"Channel": 27, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.856934, "args": {"Channel": 28, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.937012, "args": {"Channel": 29, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824049.006836, "args": {"Channel": 30, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824049.306641, "args": {"Channel": 31, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Send", "cat": "P2P", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 1818670.567871},

{"name": "Send", "cat": "P2P_API", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 1467848.663574},

{"name": "Recv", "cat": "P2P_API", "ph": "b", "id": 1, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x4142e90}},

{"name": "Recv", "cat": "P2P", "ph": "b", "id": 1, "pid": 3937375, "tid": 1, "ts": 1818658.527832, "args": {"CommHash": 12715772615376301750, "Rank": 0, "Peer": 3, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824044.566895, "args": {"Channel": 16, "StartGpuClk": 1765223116927674080, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824045.476562, "args": {"Channel": 17, "StartGpuClk": 1765223116927674080, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824045.976562, "args": {"Channel": 18, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824046.256836, "args": {"Channel": 19, "StartGpuClk": 1765223116927674112, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824046.796875, "args": {"Channel": 20, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.076660, "args": {"Channel": 21, "StartGpuClk": 1765223116927674144, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.346680, "args": {"Channel": 22, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.616699, "args": {"Channel": 23, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.756836, "args": {"Channel": 24, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824047.986816, "args": {"Channel": 25, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.256836, "args": {"Channel": 26, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.437012, "args": {"Channel": 27, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.606934, "args": {"Channel": 28, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.896973, "args": {"Channel": 29, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824048.976562, "args": {"Channel": 30, "StartGpuClk": 1765223116927674016, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1824049.176758, "args": {"Channel": 31, "StartGpuClk": 1765223116927674048, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Recv", "cat": "P2P", "ph": "e", "id": 1, "pid": 3937375, "tid": 1, "ts": 1818670.607910},

{"name": "Recv", "cat": "P2P_API", "ph": "e", "id": 1, "pid": 3937375, "tid": 1, "ts": 1467850.673828},

{"name": "Send", "cat": "P2P_API", "ph": "b", "id": 2, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x4142e50}},

{"name": "Send", "cat": "P2P", "ph": "b", "id": 2, "pid": 3937375, "tid": 1, "ts": 1813823.017578, "args": {"CommHash": 12715772615376301750, "Rank": 1, "Peer": 2, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819759.735840, "args": {"Channel": 16, "StartGpuClk": 1765223120647845760, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819760.105957, "args": {"Channel": 17, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819760.695801, "args": {"Channel": 18, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819760.965820, "args": {"Channel": 19, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.305664, "args": {"Channel": 20, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.565918, "args": {"Channel": 21, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.885742, "args": {"Channel": 22, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.155762, "args": {"Channel": 23, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.315918, "args": {"Channel": 24, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.505859, "args": {"Channel": 25, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.916016, "args": {"Channel": 26, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.115723, "args": {"Channel": 27, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.275879, "args": {"Channel": 28, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.785645, "args": {"Channel": 29, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819764.025879, "args": {"Channel": 30, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819764.275879, "args": {"Channel": 31, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Send", "cat": "P2P", "ph": "e", "id": 2, "pid": 3937375, "tid": 1, "ts": 1813965.547852},

{"name": "Send", "cat": "P2P_API", "ph": "e", "id": 2, "pid": 3937375, "tid": 1, "ts": 1467857.923828},

{"name": "Recv", "cat": "P2P_API", "ph": "b", "id": 3, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x4142e50}},

{"name": "Recv", "cat": "P2P", "ph": "b", "id": 3, "pid": 3937375, "tid": 1, "ts": 1813949.228027, "args": {"CommHash": 12715772615376301750, "Rank": 1, "Peer": 0, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819758.825684, "args": {"Channel": 16, "StartGpuClk": 1765223120647845760, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819759.965820, "args": {"Channel": 17, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819760.555664, "args": {"Channel": 18, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819760.925781, "args": {"Channel": 19, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.186035, "args": {"Channel": 20, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.436035, "args": {"Channel": 21, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819761.755859, "args": {"Channel": 22, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.115723, "args": {"Channel": 23, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.275879, "args": {"Channel": 24, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.465820, "args": {"Channel": 25, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819762.785645, "args": {"Channel": 26, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.075684, "args": {"Channel": 27, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.235840, "args": {"Channel": 28, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.425781, "args": {"Channel": 29, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819763.916016, "args": {"Channel": 30, "StartGpuClk": 1765223120647845728, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1819764.235840, "args": {"Channel": 31, "StartGpuClk": 1765223120647845696, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Recv", "cat": "P2P", "ph": "e", "id": 3, "pid": 3937375, "tid": 1, "ts": 1813965.807617},

{"name": "Recv", "cat": "P2P_API", "ph": "e", "id": 3, "pid": 3937375, "tid": 1, "ts": 1467858.523926},

{"name": "Send", "cat": "P2P_API", "ph": "b", "id": 4, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x420aa70}},

{"name": "Send", "cat": "P2P", "ph": "b", "id": 4, "pid": 3937375, "tid": 1, "ts": 1808856.147949, "args": {"CommHash": 12715772615376301750, "Rank": 2, "Peer": 3, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814752.985840, "args": {"Channel": 16, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.145996, "args": {"Channel": 17, "StartGpuClk": 1765223121869204288, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.455566, "args": {"Channel": 18, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.745605, "args": {"Channel": 19, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.015625, "args": {"Channel": 20, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.255859, "args": {"Channel": 21, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.505859, "args": {"Channel": 22, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.675781, "args": {"Channel": 23, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.916016, "args": {"Channel": 24, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.105957, "args": {"Channel": 25, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.375977, "args": {"Channel": 26, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.535645, "args": {"Channel": 27, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.715820, "args": {"Channel": 28, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.795898, "args": {"Channel": 29, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.955566, "args": {"Channel": 30, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814756.115723, "args": {"Channel": 31, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Send", "cat": "P2P", "ph": "e", "id": 4, "pid": 3937375, "tid": 1, "ts": 1808997.577637},

{"name": "Send", "cat": "P2P_API", "ph": "e", "id": 4, "pid": 3937375, "tid": 1, "ts": 1467864.973633},

{"name": "Recv", "cat": "P2P_API", "ph": "b", "id": 5, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x420aa70}},

{"name": "Recv", "cat": "P2P", "ph": "b", "id": 5, "pid": 3937375, "tid": 1, "ts": 1808980.997559, "args": {"CommHash": 12715772615376301750, "Rank": 2, "Peer": 1, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814752.325684, "args": {"Channel": 16, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.105957, "args": {"Channel": 17, "StartGpuClk": 1765223121869204288, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.275879, "args": {"Channel": 18, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.615723, "args": {"Channel": 19, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814753.905762, "args": {"Channel": 20, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.135742, "args": {"Channel": 21, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.375977, "args": {"Channel": 22, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.635742, "args": {"Channel": 23, "StartGpuClk": 1765223121869204352, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814754.795898, "args": {"Channel": 24, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.065918, "args": {"Channel": 25, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.335938, "args": {"Channel": 26, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.485840, "args": {"Channel": 27, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.575684, "args": {"Channel": 28, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.755859, "args": {"Channel": 29, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814755.925781, "args": {"Channel": 30, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1814756.075684, "args": {"Channel": 31, "StartGpuClk": 1765223121869204320, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Recv", "cat": "P2P", "ph": "e", "id": 5, "pid": 3937375, "tid": 1, "ts": 1808997.617676},

{"name": "Recv", "cat": "P2P_API", "ph": "e", "id": 5, "pid": 3937375, "tid": 1, "ts": 1467865.723633},

{"name": "Send", "cat": "P2P_API", "ph": "b", "id": 6, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x262ca50}},

{"name": "Send", "cat": "P2P", "ph": "b", "id": 6, "pid": 3937375, "tid": 1, "ts": 1804045.958008, "args": {"CommHash": 12715772615376301750, "Rank": 3, "Peer": 0, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810754.614746, "args": {"Channel": 16, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.004883, "args": {"Channel": 17, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.414551, "args": {"Channel": 18, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.634766, "args": {"Channel": 19, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.054688, "args": {"Channel": 20, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.424805, "args": {"Channel": 21, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.714844, "args": {"Channel": 22, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.895020, "args": {"Channel": 23, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.064941, "args": {"Channel": 24, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.364746, "args": {"Channel": 25, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.645020, "args": {"Channel": 26, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.804688, "args": {"Channel": 27, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.994629, "args": {"Channel": 28, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.375000, "args": {"Channel": 29, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.564941, "args": {"Channel": 30, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.814941, "args": {"Channel": 31, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Send", "cat": "P2P", "ph": "e", "id": 6, "pid": 3937375, "tid": 1, "ts": 1804217.357910},

{"name": "Send", "cat": "P2P_API", "ph": "e", "id": 6, "pid": 3937375, "tid": 1, "ts": 1467872.903809},

{"name": "Recv", "cat": "P2P_API", "ph": "b", "id": 7, "pid": 3937375, "tid": 1, "ts": 0.000000, "args": {"count": 268435456, "datatype": ncclFloat32, "GraphCaptured":0, "Stream": 0x262ca50}},

{"name": "Recv", "cat": "P2P", "ph": "b", "id": 7, "pid": 3937375, "tid": 1, "ts": 1804199.567871, "args": {"CommHash": 12715772615376301750, "Rank": 3, "Peer": 2, "Count": 268435456, "Datatype": "ncclFloat32", "nChannels": 16}},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810753.864746, "args": {"Channel": 16, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810754.744629, "args": {"Channel": 17, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.244629, "args": {"Channel": 18, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.574707, "args": {"Channel": 19, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810755.824707, "args": {"Channel": 20, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.204590, "args": {"Channel": 21, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.574707, "args": {"Channel": 22, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.764648, "args": {"Channel": 23, "StartGpuClk": 1765223112919376768, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810756.934570, "args": {"Channel": 24, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.194824, "args": {"Channel": 25, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.604980, "args": {"Channel": 26, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.774902, "args": {"Channel": 27, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810757.844727, "args": {"Channel": 28, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.224609, "args": {"Channel": 29, "StartGpuClk": 1765223112919376832, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.524902, "args": {"Channel": 30, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1810758.694824, "args": {"Channel": 31, "StartGpuClk": 1765223112919376800, "StopGpuClk": 0}},

{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 0.000000},

{"name": "Recv", "cat": "P2P", "ph": "e", "id": 7, "pid": 3937375, "tid": 1, "ts": 1804217.387695},

{"name": "Recv", "cat": "P2P_API", "ph": "e", "id": 7, "pid": 3937375, "tid": 1, "ts": 1467873.533691},

{"name": "Group API", "cat": "GROUP_API", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 1818689.007812},

{"name": "Idle", "cat": "PROXY", "ph": "b", "id": 0, "pid": 3937375, "tid": 1, "ts": 1791569.713867},

{"name": "Idle", "cat": "PROXY", "ph": "e", "id": 0, "pid": 3937375, "tid": 1, "ts": 1791571.023926},

{"name": "Sleep", "cat": "PROXY", "ph": "b", "id": 1, "pid": 3937375, "tid": 1, "ts": 1791571.173828},

{"name": "Sleep", "cat": "PROXY", "ph": "e", "id": 1, "pid": 3937375, "tid": 1, "ts": 1818675.937988},

{"name": "Append", "cat": "PROXY", "ph": "b", "id": 2, "pid": 3937375, "tid": 1, "ts": 1818677.107910, "args": {"appended": 32}},

{"name": "Append", "cat": "PROXY", "ph": "e", "id": 2, "pid": 3937375, "tid": 1, "ts": 1823927.756836},

{"name": "Idle", "cat": "PROXY", "ph": "b", "id": 3, "pid": 3937375, "tid": 1, "ts": 1824049.396973},

{"name": "Idle", "cat": "PROXY", "ph": "e", "id": 3, "pid": 3937375, "tid": 1, "ts": 1824049.636719},

{"name": "Sleep", "cat": "PROXY", "ph": "b", "id": 4, "pid": 3937375, "tid": 1, "ts": 1827149.590820},

{"name": "Sleep", "cat": "PROXY", "ph": "e", "id": 4, "pid": 3937375, "tid": 1, "ts": 2354325.189941},

{"name": "Append", "cat": "PROXY", "ph": "b", "id": 5, "pid": 3937375, "tid": 1, "ts": 2354328.969727, "args": {"appended": 0}},

{"name": "Append", "cat": "PROXY", "ph": "e", "id": 5, "pid": 3937375, "tid": 1, "ts": 2354329.399902},

{}]



i ran the @run_profiler_02_point_to_point.sh slurm script in my hpc environment. this is the json trace dumped from the nccl example profiler. what do each of these field mean?

---

**Cursor**


Reviewing the codebase and documentation to explain the trace fields.



Reading the example profiler README for more details:



The trace uses Chrome Trace Event Format (JSON). Field meanings:

## Common fields (all events)

- `name`: Event name (e.g., "Send", "Recv", "KernelCh", "Group API")
- `cat`: Category (e.g., "P2P_API", "P2P", "GPU", "GROUP_API", "KERNEL_LAUNCH", "PROXY")
- `ph`: Phase
  - `"b"` = begin
  - `"e"` = end
- `id`: Unique event ID (used to match begin/end pairs)
- `pid`: Process ID (3937375 in your trace)
- `tid`: Thread ID (1 in your trace)
- `ts`: Timestamp in microseconds (relative to trace start)
- `args`: Event-specific arguments (varies by event type)

## Event-specific `args` fields

### Group API events
- `groupApiId`: Group API identifier
- `groupDepth`: Nesting depth of group operations

### KernelLaunch events
- `groupId`: Associated group ID
- `Stream`: CUDA stream pointer

### P2P_API events (Send/Recv API calls)
- `count`: Number of elements
- `datatype`: NCCL datatype (e.g., "ncclFloat32")
- `GraphCaptured`: Whether captured in CUDA graph (0/1)
- `Stream`: CUDA stream pointer

### P2P events (actual Send/Recv operations)
- `CommHash`: Communicator unique identifier
- `Rank`: NCCL rank
- `Peer`: Peer rank
- `Count`: Number of elements
- `Datatype`: NCCL datatype
- `nChannels`: Number of channels used

### KernelCh events (GPU kernel channel operations)
- `Channel`: Channel number (16-31 in your trace)
- `StartGpuClk`: GPU clock timestamp at start
- `StopGpuClk`: GPU clock timestamp at end (0 if not yet completed)

### PROXY events
- `Channel`: Channel number
- `Peer`: Peer rank
- `Steps`: Number of network steps
- `ChunkSize`: Size of each chunk
- `transSize`: Total transfer size
- `appended`: Number of operations appended (for Append events)

## Your trace summary

Your trace shows a ring pattern with 4 GPUs (ranks 0-3):
1. Group API: Wraps the entire operation
2. KernelLaunch: CUDA kernel launches for each GPU
3. Send/Recv operations: Point-to-point communication in a ring
4. KernelCh: Individual channel operations (16 channels per GPU, channels 16-31)
5. PROXY events: Proxy thread activity (Idle, Sleep, Append)

The timestamps (`ts`) are in microseconds and show the execution timeline. You can visualize this trace in Chrome by opening `chrome://tracing` and loading the JSON file.

---

**User**

does this trace also show when the communicator was created? iirc the profiler plugin init function is called in init.cc (an llm said so previously according to @cursor_plugin_api_notification_on_communicators_and_configurations.md )

and so i guess this init function could be tracked in the trace and mark the relative "zero" timestamp...? but is it tracked?

---

**Cursor**

Looking at the example profiler code, I can see that:

## The `startTime` is set during `init`, but it's NOT tracked as an event

In `ext-profiler/example/plugin.cc`:

```126:126:ext-profiler/example/plugin.cc
    startTime = gettime();
```

This happens on line 126 when the **first** communicator's `init` is called. All subsequent timestamps are computed as `gettime() - startTime`, making this the relative "zero" point.

## However, the `init` call itself is NOT recorded in the trace

Looking at the trace output, there's no `"Init"` or `"CommCreated"` event. The trace only contains:
- `GROUP_API` events
- `KERNEL_LAUNCH` events  
- `P2P_API` / `P2P` events
- `GPU` (KernelCh) events
- `PROXY` events

The `init` function just:
1. Sets up the `startTime` (on first communicator only)
2. Allocates memory pools
3. Stores communicator metadata (`commHash`, `rank`, `nranks`)
4. Logs to NCCL's debug system (line 141: `INFO(NCCL_INIT, ...)`)

But it doesn't create a trace event entry.

## Why some timestamps show `0.000000`

In your trace, you'll notice many events have `"ts": 0.000000` - this is because those are **begin** (`"ph": "b"`) events where the start time wasn't recorded (they get their timing from the corresponding **end** event). This is a quirk of how the example profiler handles certain event types.

## Could it be tracked?

Yes! You could modify the example profiler to add an "Init" event in `exampleProfilerInit`:

```c
// After line 126, you could add:
fprintf(some_log, "{\"name\": \"CommInit\", \"cat\": \"INIT\", \"ph\": \"i\", \"pid\": %d, \"tid\": 1, \"ts\": 0.000000, \"args\": {\"commHash\": %lu, \"rank\": %d, \"nranks\": %d}},\n",
        getpid(), commId, rank, nranks);
```

But the current implementation doesn't do this - it only starts tracing after `init` completes, so all traced events are relative to when profiling started, not when the communicator was actually created.

```plaintext
    startTime = gettime();
```

```c
// After line 126, you could add:
fprintf(some_log, "{\"name\": \"CommInit\", \"cat\": \"INIT\", \"ph\": \"i\", \"pid\": %d, \"tid\": 1, \"ts\": 0.000000, \"args\": {\"commHash\": %lu, \"rank\": %d, \"nranks\": %d}},\n",
        getpid(), commId, rank, nranks);
```

---

**User**

ok. since init marks the starttime and init is when the communicator is created, it is still implicitly tracked. and the destruction (probaly happens as the plugin api's finalize function is called?) is not tracked either? is this implicitly known when the destruction happened or not?

---

**Cursor**

You're right that `init` implicitly marks time zero. Let me check what happens at `finalize` (communicator destruction):

## Destruction is NOT tracked - not even implicitly

Looking at `exampleProfilerFinalize`:

```195:249:ext-profiler/example/plugin.cc
__hidden ncclResult_t exampleProfilerFinalize(void* context) {
  FILE* fh = NULL;
  // ...
  // Just dumps events and frees memory - NO timestamp recorded!
  // ...
  if (fh) fprintf(fh, "{}]\n");
  if (fh) fclose(fh);
  return ncclSuccess;
}
```

The `finalize` function:
1. Opens the output file
2. Dumps all collected events from the pools
3. Frees memory
4. Closes the file

**But it does NOT:**
- Record a timestamp for when destruction happened
- Create a "CommDestroyed" or "Finalize" event in the trace

## Summary: What's tracked implicitly vs explicitly

| Event | Explicitly Tracked | Implicitly Known |
|-------|-------------------|------------------|
| Communicator creation (`init`) | ❌ No event | ✅ Yes - it's `ts = 0` (the reference point) |
| Communicator destruction (`finalize`) | ❌ No event | ❌ **No** - not known at all |

## How you *could* infer destruction time (approximately)

1. **Last event timestamp**: The maximum `ts` value in the trace is *close to* but not exactly when destruction happened (there could be a gap between the last operation and `ncclCommDestroy`)

2. **File modification time**: The JSON file's modification timestamp is approximately when `finalize` ran

3. **Modify the plugin**: Add a finalize event:
```c
// In exampleProfilerFinalize, before closing the file:
fprintf(fh, "{\"name\": \"CommDestroy\", \"cat\": \"LIFECYCLE\", \"ph\": \"i\", \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": {\"commHash\": %lu}},\n",
        getpid(), gettime() - startTime, ctx->commHash);
```

But as-is, the current profiler doesn't give you destruction time.

```plaintext
__hidden ncclResult_t exampleProfilerFinalize(void* context) {
  FILE* fh = NULL;
  // ...
  // Just dumps events and frees memory - NO timestamp recorded!
  // ...
  if (fh) fprintf(fh, "{}]\n");
  if (fh) fclose(fh);
  return ncclSuccess;
}
```

```c
// In exampleProfilerFinalize, before closing the file:
fprintf(fh, "{\"name\": \"CommDestroy\", \"cat\": \"LIFECYCLE\", \"ph\": \"i\", \"pid\": %d, \"tid\": 1, \"ts\": %f, \"args\": {\"commHash\": %lu}},\n",
        getpid(), gettime() - startTime, ctx->commHash);
```

