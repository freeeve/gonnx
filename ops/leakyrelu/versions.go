package leakyrelu

import "github.com/advancedclimatesystems/gonnx/ops"

var leakyReluVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newLeakyRelu, 6, leakyReluTypeConstraints),
	16: ops.NewOperatorConstructor(newLeakyRelu, 16, leakyReluTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return leakyReluVersions
}
