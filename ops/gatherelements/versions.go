package gatherelements

import "github.com/advancedclimatesystems/gonnx/ops"

var gatherElementsVersions = ops.OperatorVersions{
	11: ops.NewOperatorConstructor(newGatherElements, 11, gatherElementsTypeConstraints),
	13: ops.NewOperatorConstructor(newGatherElements, 13, gatherElementsTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return gatherElementsVersions
}
