package reducesum

import "github.com/advancedclimatesystems/gonnx/ops"

var reduceSumVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newReduceSum, 1, reduceSumTypeConstraints),
	11: ops.NewOperatorConstructor(newReduceSum, 11, reduceSumTypeConstraints),
	13: ops.NewOperatorConstructor(newReduceSum13, 13, reduceSumV13TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return reduceSumVersions
}
